"""Local, bounded RoboTwin/WAM IPC. Importing this module never imports torch/Scene.

Only the worker's explicit main constructs a model. JSON + uint8 RGB base64,
not pickle, crosses the simulator NumPy1 / training NumPy2 boundary.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("head_camera", "left_camera", "right_camera")
MODEL_VIEWS = ("observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0",
               "observation.rgb.right_realsense_link_camera_0")
MAX_MESSAGE = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 1920 * 1080 * 3
PROTOCOL = 1
TASK = "put_object_cabinet"  # Legacy default; each run is bound to its own manifest task.


def contract_task(config):
    task = config["dataset"]["task_name"]
    if (not isinstance(task, str) or not task
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in task)):
        raise InfrastructureError("invalid training task identifier")
    return task


class InfrastructureError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path, content=True):
    p = Path(path).resolve()
    s = p.stat()
    result = {"path": str(p), "size": s.st_size, "mtime_ns": s.st_mtime_ns}
    if content:
        result["sha256"] = sha256(p)
    return result


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def strict_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise InfrastructureError(f"protocol keys differ: expected {sorted(keys)}")


def vector(value, n, label):
    import numpy as np
    x = np.asarray(value)
    if x.shape != (n,) or x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise InfrastructureError(f"{label}: expected finite real ({n},)")
    return x.copy()


def ids(episode, step):
    if not isinstance(episode, str) or not 1 <= len(episode) <= 128:
        raise InfrastructureError("invalid episode ID")
    if type(step) is not int or not 0 <= step <= 100000:
        raise InfrastructureError("invalid step ID")


def envelope(kind, episode, step, **payload):
    ids(episode, step)
    return {"version": PROTOCOL, "kind": kind, "episode": episode, "step": step, **payload}


def check_reply(message, kind, episode, step):
    if (not isinstance(message, dict) or message.get("version") != PROTOCOL
            or message.get("kind") != kind or message.get("episode") != episode
            or type(message.get("step")) is not int or message["step"] != step):
        raise InfrastructureError(f"wrong worker reply/ID, expected {kind}/{episode}/{step}: {str(message)[:400]}")


def encode_observation(observation):
    """Accept only the adapter's result, never native get_obs or evaluation GT."""
    import numpy as np
    strict_keys(observation, ("rgb", "commanded_joint_gripper_state"))
    strict_keys(observation["rgb"], VIEWS)
    state = vector(observation["commanded_joint_gripper_state"], 14, "commanded state")
    rgb = {}
    for view in VIEWS:
        image = np.asarray(observation["rgb"][view])
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
                or min(image.shape[:2]) <= 0 or image.nbytes > MAX_IMAGE_BYTES):
            raise InfrastructureError("RGB must be bounded uint8 HWC3")
        rgb[view] = {"shape": list(image.shape), "bytes": base64.b64encode(image.tobytes()).decode("ascii")}
    return {"rgb": rgb, "commanded_joint_gripper_state": state.tolist()}


def decode_observation(value):
    import numpy as np
    strict_keys(value, ("rgb", "commanded_joint_gripper_state"))
    strict_keys(value["rgb"], VIEWS)
    result = {"rgb": {}, "commanded_joint_gripper_state": vector(value["commanded_joint_gripper_state"], 14, "state")}
    for view in VIEWS:
        item = value["rgb"][view]
        strict_keys(item, ("shape", "bytes"))
        shape = item["shape"]
        if (not isinstance(shape, list) or len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape)
                or shape[2] != 3 or math.prod(shape) > MAX_IMAGE_BYTES
                or not isinstance(item["bytes"], str) or len(item["bytes"]) > 4 * ((MAX_IMAGE_BYTES + 2) // 3)):
            raise InfrastructureError("invalid RGB geometry/size")
        try:
            raw = base64.b64decode(item["bytes"], validate=True)
        except (ValueError, TypeError) as exc:
            raise InfrastructureError("invalid RGB base64") from exc
        if len(raw) != math.prod(shape):
            raise InfrastructureError("RGB byte count mismatch")
        result["rgb"][view] = np.frombuffer(raw, np.uint8).reshape(shape).copy()
    return result


class Channel:
    """A single deadline covers the entire frame, including partial reads/writes."""
    def __init__(self, sock):
        self.sock = sock

    def _read(self, size, deadline):
        chunks = bytearray()
        while len(chunks) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InfrastructureError("worker receive timeout")
            self.sock.settimeout(remaining)
            block = self.sock.recv(min(size - len(chunks), 65536))
            if not block:
                raise InfrastructureError("worker closed IPC/terminated")
            chunks.extend(block)
        return bytes(chunks)

    def recv(self, timeout):
        try:
            deadline = time.monotonic() + timeout
            length, = struct.unpack("!I", self._read(4, deadline))
            if not 1 <= length <= MAX_MESSAGE:
                raise InfrastructureError("oversized/empty IPC frame")
            value = json.loads(self._read(length, deadline), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if not isinstance(value, dict):
                raise ValueError("expected object")
            return value
        except (OSError, ValueError, UnicodeError) as exc:
            raise InfrastructureError(f"worker receive: {exc}") from exc

    def send(self, value, timeout):
        try:
            data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            if not 1 <= len(data) <= MAX_MESSAGE:
                raise InfrastructureError("oversized IPC frame")
            self.sock.settimeout(timeout)
            self.sock.sendall(struct.pack("!I", len(data)) + data)
        except (OSError, ValueError, TypeError) as exc:
            raise InfrastructureError(f"worker send: {exc}") from exc


def verify_gpu_owner(expected):
    from scripts.gpu_lock import owner
    if not expected or owner() != expected:
        raise InfrastructureError("matching externally held --gpu-lock-owner is required; lock is never acquired/released here")
    # A matching trainer owner is NOT permission to overlap the training job.
    trainers = {"train_robotwin_wam.py", "train_wam_v3.py", "train_wam_task_v3.py", "train_lilawam_behavior.py"}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(Path(arg).name in trainers for arg in argv if arg):
            raise InfrastructureError(f"training process {entry.name} still present; never share its GPU lock")


def training_contract(run_dir, checkpoint):
    """CPU metadata checks. Actual checkpoint tensors/content are checked by worker."""
    run, ck = Path(run_dir).resolve(), Path(checkpoint).resolve()
    if ck.parent != run or not ck.is_file():
        raise InfrastructureError("checkpoint must be an explicit local file in training-run-dir")
    manifest = json.loads((run / "manifest.json").read_text())
    contract = manifest["contract"]
    if manifest["contract_sha256"] != digest(contract) or contract["format"] != "robotwin-native14-v1":
        raise InfrastructureError("training manifest format/digest mismatch")
    cfg = contract["config"]
    task = contract_task(cfg)
    if (cfg["common"]["action_dim"] != 14
            or cfg["common"]["state_dim"] != 14 or cfg["common"]["action_chunk_size"] != 32
            or cfg["common"].get("action_execution_horizon") != 8 or cfg["common"]["num_inference_steps"] != 10
            or cfg["dataset"]["cond_mode"] != "zero_single_task" or tuple(cfg["model"]["views"]) != MODEL_VIEWS):
        raise InfrastructureError("task/native14/K1-Euler10/chunk32-exec8/three-view contract mismatch")
    for path, expected in contract["source_hashes"].items():
        if sha256(path) != expected:
            raise InfrastructureError(f"frozen training source differs: {path}")
    cache_path = Path(contract["cache_config"]["path"])
    if sha256(cache_path) != contract["cache_config"]["sha256"]:
        raise InfrastructureError("cache config changed")
    cache = json.loads(cache_path.read_text())
    if cache["source"]["task"] != task:
        raise InfrastructureError("cache task mismatch")
    split = ROOT / cfg["dataset"]["split_manifest"]
    if sha256(split) != cfg["dataset"]["split_sha256"]:
        raise InfrastructureError("frozen dataset split differs")
    stats = {}
    for name in ("normalization", "temporal_stats"):
        p = run / f"{name}.json"
        if sha256(p) != contract["stats"][name]["sha256"]:
            raise InfrastructureError(f"TRAIN {name} bytes differ")
        stats[name] = json.loads(p.read_text())
        provenance = stats[name]["provenance"]
        if provenance["fit_split"] != "train" or provenance["episodes"] != contract["episodes"]["train"]:
            raise InfrastructureError("statistics not fitted on this run's TRAIN episodes")
    receipt = json.loads(ck.with_name(ck.name + ".sha256.json").read_text())
    if receipt["size"] != ck.stat().st_size or receipt.get("model_only") is not True:
        raise InfrastructureError("use an immutable model-only milestone checkpoint, not mutable checkpoint_last")
    return contract, cache, stats, receipt


class PolicyClient:
    """Owns only its worker process group; constructor failure also reaps it.

    command is an explicit dependency-injection seam for CPU unit tests, not a
    CLI option or production fallback. stderr/stdout go to the run's worker log.
    """
    def __init__(self, command, log_path, startup_timeout=180, timeout=30):
        self.proc = self.channel = self.log = None
        self.timeout = timeout
        parent, child = socket.socketpair()
        try:
            self.log = Path(log_path).open("xb")
            env = os.environ.copy()
            env.update({k: "2" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
            self.proc = subprocess.Popen([*command, "--ipc-fd", str(child.fileno())], pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log, start_new_session=True, env=env, cwd=ROOT)
            child.close()
            self.channel = Channel(parent)
            self.ready = self.channel.recv(startup_timeout)
            check_reply(self.ready, "ready", "startup", 0)
            self.check_alive()
        except BaseException:
            parent.close()
            child.close()
            self.close()
            raise

    def check_alive(self):
        if self.proc is None or self.proc.poll() is not None:
            raise InfrastructureError("policy worker exited")

    def request(self, message, kind):
        try:
            self.check_alive()
            self.channel.send(message, self.timeout)
            reply = self.channel.recv(self.timeout)
            check_reply(reply, kind, message["episode"], message["step"])
            self.check_alive()
            return reply
        except BaseException:
            self.close()
            raise

    def reset(self, episode, seed):
        return self.request(envelope("reset", episode, 0, seed=seed), "reset_ok")

    def clear_actions(self, episode, next_step, reason):
        return self.request(envelope("clear_actions", episode, next_step, reason=reason), "cleared")

    def action(self, episode, step, observation):
        try:
            reply = self.request(envelope("observe", episode, step, observation=encode_observation(observation)), "action")
            vector(reply["raw_action"], 14, "worker action")
            return reply
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.channel is not None:
            self.channel.sock.close()
            self.channel = None
        if self.proc is not None:
            # killpg targets only the session we created, even if its leader exited.
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=2)
            self.proc = None
        if self.log is not None:
            self.log.close()
            self.log = None


class ChunkPolicy:
    """Encode+push every observation; only sampling is skipped while queued."""
    def __init__(self, context, encode, sample, execution_mode="native8"):
        if execution_mode not in ("native8", "contact_refresh"):
            raise ValueError("unsupported execution mode")
        self.execution_mode = execution_mode
        self.execution_prefix = 8
        self.execution_reason = "native_exec8"
        self.context, self.encode, self.sample = context, encode, sample
        self.queue = deque()
        self.episode = None
        self.next_step = 0

    def reset(self, episode, seed):
        ids(episode, 0)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise InfrastructureError("invalid seed")
        self.context.reset()
        self.queue.clear()
        self.episode, self.next_step, self.seed = episode, 0, seed

    def clear_actions(self, episode, next_step, reason):
        if episode != self.episode or type(next_step) is not int or next_step != self.next_step:
            raise InfrastructureError("queue invalidation episode/step mismatch")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 256:
            raise InfrastructureError("bounded nonempty replan reason required")
        discarded = len(self.queue)
        self.queue.clear()
        return {"discarded_actions": discarded, "history_preserved": True, "next_step": self.next_step}

    def observe(self, episode, step, observation):
        if episode != self.episode or step != self.next_step:
            raise InfrastructureError("stale or out-of-order observation")
        features, pool = self.encode(observation["rgb"])
        self.context.push(observation["commanded_joint_gripper_state"], pool)
        replanned = not self.queue
        if replanned:
            import numpy as np
            chunk = np.asarray(self.sample(features, self.context.inputs(), self.seed, step))
            if chunk.shape != (32, 14) or chunk.dtype.kind not in "fiu" or not np.isfinite(chunk).all():
                raise InfrastructureError("expected a finite 32x14 action chunk")
            from harness.robotwin_observation_schedule import execution_prefix
            self.execution_prefix, self.execution_reason = execution_prefix(chunk, observation["commanded_joint_gripper_state"], self.execution_mode)
            self.queue.extend(row.copy() for row in chunk[:self.execution_prefix])
        action = vector(self.queue.popleft(), 14, "sampled action")
        self.next_step += 1
        return {"raw_action": action.tolist(), "replanned": replanned, "history_steps": self.next_step,
                "queue_remaining": len(self.queue), "execution_prefix": self.execution_prefix, "execution_reason": self.execution_reason}


class WamPolicy(ChunkPolicy):
    def __init__(self, run_dir, checkpoint, gpu_owner, execution_mode="native8", target_head=None, vtt_adapter=None, vtt_bank=None):
        from harness.robotwin_eval_evidence import ResourceGuard
        self.guard = ResourceGuard()
        self.guard.check()
        verify_gpu_owner(gpu_owner)
        contract, cache, stats, receipt = training_contract(run_dir, checkpoint)
        if sha256(checkpoint) != receipt["sha256"]:
            raise InfrastructureError("checkpoint content/receipt mismatch")
        import cv2
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from harness.robotwin_training_data import preprocess_rgb, cache_identity
        from harness.wam_behavior import load_upstream
        from harness.wam_v3 import create_v3_action_model, WamV3Wrapper, V3Context
        cfg = contract["config"]
        # Uses the exact preprocessing implementation AND NumPy/OpenCV ABI from cache.
        actual_cache = cache_identity(json.loads((ROOT / cfg["dataset"]["split_manifest"]).read_text()))
        if actual_cache != cache:
            raise InfrastructureError("training RGB/cache ABI or local encoder provenance differs")
        encoder_path = (ROOT / cfg["model"]["vision_encoder"]["checkpoint_path"]).resolve()
        if sha256(encoder_path / "model.safetensors") != cache["encoder_sha256"]:
            raise InfrastructureError("local frozen DINO content mismatch")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        cv2.setNumThreads(2)
        # Trusted local training checkpoints only. No pickle ever crosses IPC.
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if (ck.get("format") != "robotwin-native14-v1" or ck.get("contract") != contract
                or ck.get("contract_sha256") != digest(contract) or ck.get("step") != receipt["step"]
                or ck.get("model_only") is not True):
            raise InfrastructureError("checkpoint/run contract mismatch")
        verify_gpu_owner(gpu_owner)
        model = create_v3_action_model(OmegaConf.create(cfg), dino_hidden_size=1024,
            num_dino_layers=3, task_cond_dim=1024, patch_size=16)
        # Deliberately NOT load_retargeted_v3_state: all six projection tensors load.
        model.load_state_dict(ck["model_state_dict"], strict=True)
        del ck
        self.task_condition = np.zeros(1024, np.float32)
        vtt_metadata = None
        self.vtt_monitor = None
        self.vtt_observation = None
        if vtt_adapter:
            from harness.robotwin_vtt_runtime import load_adapter
            self.task_condition, vtt_metadata = load_adapter(vtt_adapter, model, receipt['sha256'])
        if vtt_bank:
            from harness.robotwin_vtt_runtime import VTTMonitor
            self.vtt_monitor = VTTMonitor(vtt_bank)
        self.condition_source = 'trained_VTT_prototype' if vtt_adapter else 'zero_single_task'
        self.condition_library = None
        self.initial_cls = None
        if vtt_adapter and vtt_bank:
            from harness.robotwin_condition_memory import DemoConditionMemory
            self.condition_library = DemoConditionMemory(vtt_bank, vtt_adapter)
        _, runner = load_upstream()
        encoder, hidden, registers, patch = runner.ModelFactory.create_vision_encoder(
            str(encoder_path), torch.bfloat16, torch.device("cuda"))
        if (hidden, registers, patch) != (1024, 4, 16):
            raise InfrastructureError("unexpected DINO geometry")
        encoder.requires_grad_(False).eval()
        wrapper = WamV3Wrapper(model, views=cfg["model"]["views"], stats=stats["temporal_stats"],
            norm_stats_path=Path(run_dir) / "normalization.json", normalization_key="robotwin",
            proprio_mask="native_all14", proprio_keep_override=np.ones(14), train_cfg=cfg["training"],
            vision_encoder=encoder, feat_layers=(-12, -8, -4), num_register_tokens=4, device="cuda").to("cuda").eval()
        self.wrapper = wrapper
        self.runtime = {"numpy": np.__version__, "opencv": cv2.__version__, "torch": str(torch.__version__),
            "checkpoint_sha256": receipt["sha256"], "checkpoint_step": receipt["step"],
            "contract_sha256": digest(contract), "encoder_sha256": cache["encoder_sha256"],
            "strict_load": True, "preprocess_sha256": cache["preprocess_sha256"], "device": "cuda"}
        self.runtime['vtt_adapter'] = vtt_metadata
        self.runtime['vtt_monitor'] = self.vtt_monitor.identity if self.vtt_monitor else None
        target_net=None
        self.target_observation=None
        if target_head:
            from harness.robotwin_wrist_target import model as target_model
            target_ck=torch.load(target_head,map_location='cpu',weights_only=False)
            if target_ck.get('format')!='wrist_target_patch_mlp_v1':raise InfrastructureError('unsupported_target_head_format')
            target_net=target_model();target_net.load_state_dict(target_ck['state_dict'],strict=True);target_net.to('cuda').eval().requires_grad_(False)
            self.runtime['target_head']={'sha256':sha256(target_head),'status':'advisory_unadmitted','no_physical_verification':True}

        def encode(rgb):
            verify_gpu_owner(gpu_owner)
            self.guard.check()
            with torch.inference_mode():
                pixels = np.stack([preprocess_rgb(rgb[view]) for view in VIEWS])
                feats, pool = wrapper.encode(torch.from_numpy(pixels)[None])
                cls = feats[-1][0, 0, 0].float().cpu().numpy()
                if self.initial_cls is None:
                    self.initial_cls = cls.copy()
                if self.vtt_monitor:
                    self.vtt_observation = self.vtt_monitor.observe(cls, self.next_step)
                if target_net is not None:
                    from harness.robotwin_wrist_target import patch_features,detect
                    candidates=[]
                    for v in (1,2):
                        patches=feats[-1][0,v,5:]
                        cand,reason=detect(target_net(patch_features(patches)).sigmoid().float().cpu().numpy())
                        candidates.append({'view':VIEWS[v],'candidate':cand,'reason':reason,'rgb_sha256':hashlib.sha256(rgb[VIEWS[v]].tobytes()).hexdigest()})
                    self.target_observation={'status':'advisory_unadmitted','model_sha256':self.runtime['target_head']['sha256'],'views':candidates,'grasp_verified':None}
                return feats, pool[0].float().cpu().numpy()

        def sample(feats, inputs, seed, step):
            state, pool = inputs
            sample_seed = int(hashlib.sha256(f"{seed}:{step}".encode()).hexdigest()[:15], 16)
            generator = torch.Generator(device="cuda").manual_seed(sample_seed)
            with torch.inference_mode():
                actions, _ = wrapper.sample_actions(feats, torch.from_numpy(state)[None], torch.from_numpy(pool)[None],
                    torch.from_numpy(self.task_condition.copy())[None], num_steps=10, generator=generator)
                return actions[0].float().cpu().numpy()
        v3 = cfg["model"]["v3"]
        self.guard.check()
        super().__init__(V3Context(v3["proprio_lags"], v3["memory_lags"], MODEL_VIEWS, state_dim=14), encode, sample, execution_mode=execution_mode)

    def reset(self, episode, seed):
        super().reset(episode, seed)
        self.initial_cls = None
        self.vtt_observation = None
        if self.vtt_monitor:
            self.vtt_monitor.reset()
        if self.condition_library:
            self.condition_library.reset()
            self.task_condition = self.condition_library.prototype.copy()
            self.condition_source = 'trained_VTT_prototype'

    def next_demo_condition(self, episode, step):
        if (episode != self.episode or step != self.next_step or self.queue
                or self.condition_library is None or self.initial_cls is None):
            raise InfrastructureError('condition switch requires ACKed empty queue and trained TRAIN bank')
        condition, source = self.condition_library.select(self.initial_cls)
        self.task_condition = condition.copy()
        self.condition_source = source
        return {'condition_source': source, 'condition_sha256': hashlib.sha256(condition.tobytes()).hexdigest(),
                'history_preserved': True, 'next_step': self.next_step, 'task_success_claimed': False}

    def observe(self, episode, step, observation):
        result = super().observe(episode, step, observation)
        result['task_condition'] = {'source': self.condition_source,
            'sha256': hashlib.sha256(self.task_condition.tobytes()).hexdigest()}
        if self.vtt_observation is not None:
            result['vtt'] = {**self.vtt_observation, 'episode': episode, 'observation_step': step}
        if self.target_observation is not None:
            result['target_observation']={**self.target_observation,'episode':episode,'observation_step':step}
        import torch
        import resource
        result["resources"] = {**self.guard.check(),
            "policy_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        return result


def serve(channel, policy):
    channel.send(envelope("ready", "startup", 0, runtime=policy.runtime, pid=os.getpid()), 10)
    while True:
        message = channel.recv(600)
        episode, step = message.get("episode"), message.get("step")
        ids(episode, step)
        if message.get("version") != PROTOCOL:
            raise InfrastructureError("unsupported protocol")
        started = time.monotonic()
        if message.get("kind") == "reset":
            strict_keys(message, ("version", "kind", "episode", "step", "seed"))
            if step != 0:
                raise InfrastructureError("reset step must be zero")
            policy.reset(episode, message["seed"])
            reply = envelope("reset_ok", episode, step)
        elif message.get("kind") == "clear_actions":
            strict_keys(message, ("version", "kind", "episode", "step", "reason"))
            result = policy.clear_actions(episode, step, message["reason"])
            reply = envelope("cleared", episode, step, **result)
        elif message.get('kind') == 'next_demo_condition':
            strict_keys(message, ('version', 'kind', 'episode', 'step'))
            reply = envelope('condition_selected', episode, step, **policy.next_demo_condition(episode, step))
        elif message.get("kind") == "observe":
            strict_keys(message, ("version", "kind", "episode", "step", "observation"))
            result = policy.observe(episode, step, decode_observation(message["observation"]))
            reply = envelope("action", episode, step, **result, policy_seconds=time.monotonic() - started)
        else:
            raise InfrastructureError("unknown worker request")
        channel.send(reply, 30)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ipc-fd", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--training-run-dir", required=True)
    p.add_argument("--target-head", help="optional appearance localization head; advisory only")
    p.add_argument('--vtt-adapter', help='explicit validated trained residual adapter')
    p.add_argument('--vtt-bank', help='TRAIN monitoring and demonstration condition library')
    p.add_argument("--gpu-lock-owner", required=True)
    p.add_argument("--execution-mode", choices=("native8", "contact_refresh"), default="native8")
    args = p.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    channel = Channel(socket.socket(fileno=args.ipc_fd))
    try:
        serve(channel, WamPolicy(args.training_run_dir, args.checkpoint, args.gpu_lock_owner, args.execution_mode, args.target_head, args.vtt_adapter, args.vtt_bank))
    except BaseException as exc:
        try:
            channel.send(envelope("infrastructure_error", "startup", 0, error=f"{type(exc).__name__}: {exc}"), 1)
        except BaseException:
            pass
        print(f"infrastructure_error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        channel.sock.close()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
