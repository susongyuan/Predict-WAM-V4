"""Memory over realized VTTs: promote verified skill transitions into the evidence ledger and export them as
conditioning vectors for the frozen WAM (memory writes the condition space, not the weights).

Training used instance VTTs with probability `instance_vtt_prob` (0.2). Runtime VTTs share this
representation; their distribution and transfer still require validation.

Rules (see docs/system_design.md §3.5/§4 and docs/evaluation_protocol.md):
- VTT closure alone cannot verify success: a separate reviewed postcondition label is required;
  the verifier kind is recorded and simulator truth
  (`simulator_predicates`) is never exported for the policy path (`policy_only=True` in recall);
- one SQLite ledger per split/run (harness.core.Memory); export takes the mean of verified-success VTTs per
  skill/object contract within an exact scope (task, scene fingerprint, robot/controller/WAM/VLM versions);
- nothing here changes model weights.

    python3 -m harness.memory_vtt promote <run_dir> --db <ledger.sqlite> --split development --run <run_id> \
        --verifier perception_predicates --scene-context '{"task":"loading_the_car","scene":"house_double_floor_lower"}'
    python3 -m harness.memory_vtt export --db <ledger.sqlite> --split development --run <run_id> \
        --stores <run_dir>/memory_vtt_store.jsonl --out configs/memory_vtt_<run>.json

Review file: closure_labels.json = {"<episode>": {"<index>@<end_step>": "done"|"not_done"}}.
The legacy "<index>" key is accepted only when there is a single attempt at that index.
Labels must come from reviewing observations/postconditions, not by copying the closure verdict.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from harness.core import Memory, Scope, context_fingerprint
from harness.grounding import Catalog, Step

ROOT = Path(__file__).resolve().parents[1]


def validate_vector(vector, dimension=1024):
    """Conditions cross a file boundary: reject wrong shapes, NaN/Inf and degenerate transitions."""
    v = np.asarray(vector, dtype=np.float32)
    if v.shape != (dimension,) or not np.isfinite(v).all() or float(np.linalg.norm(v)) < 1e-3:
        raise ValueError(f"Expected a finite, nonzero {dimension}D VTT")
    return v


def validate_context(actual, expected):
    """Every scope field except the skill must match the explicitly configured runtime context."""
    for context in (actual, expected):
        try:
            Scope(skill_id="context_check", **context).key()
        except (TypeError, ValueError) as exc:
            raise ValueError("A complete memory context (Scope without skill_id) is required") from exc
    if actual != expected:
        raise ValueError("Memory task/scene/model/split/run context does not match")


def _step(row):
    if "skill" in row and "objects" in row:
        step = Step(row["skill"], tuple(row["objects"]))
    else:  # older orchestrator logs store only the canonical contract key
        skill, objects = row["key"].removesuffix(")").split("(", 1)
        step = Step(skill, tuple(objects.split(",")) if objects else ())
    if "key" in row and row["key"] != step.key():
        raise ValueError("Realized VTT skill/objects disagree with its contract key")
    return step


def _scope(skill, args, versions):
    return Scope(task_id=args.task, skill_id=skill, context_fingerprint=context_fingerprint(json.loads(args.scene_context)),
                 robot_version=versions["robot"], controller_version=versions["controller"], wam_version=versions["wam"],
                 vlm_version=versions["vlm"], split_id=args.split, run_id=args.run)


def promote(args):
    run = Path(args.run_dir)
    versions = json.loads(Path(args.versions).read_text()) if args.versions else {
        "robot": "r1pro-v3.9.2", "controller": "official-r1pro.yaml", "wam": args.wam_version, "vlm": "none"}
    vtt_store = run / "memory_vtt_store.jsonl"
    n_ok = n_skip = 0
    labels = json.loads((run / "closure_labels.json").read_text()) if (run / "closure_labels.json").exists() else {}
    if args.verifier != "perception_predicates":
        raise ValueError("Only reviewed perception predicates may enter this policy memory")
    rows = [json.loads(line) for line in (run / "realized_vtt.jsonl").read_text().splitlines() if line.strip()]
    attempts = Counter((r["episode"], r["index"]) for r in rows)
    seen = {json.loads(line).get("source") for line in vtt_store.read_text().splitlines()} if vtt_store.exists() else set()
    catalog = Catalog()
    memory = Memory(args.db, split_id=args.split, run_id=args.run)
    try:
        with vtt_store.open("a") as store:
            for r in rows:
                source = f"{run.resolve()}/realized_vtt.jsonl#{r['episode']}:{r['index']}@{r['end_step']}"
                if source in seen:
                    continue
                episode_labels = labels.get(str(r["episode"]), {})
                reviewed = episode_labels.get(f"{r['index']}@{r['end_step']}")
                if reviewed is None and attempts[r["episode"], r["index"]] == 1:
                    reviewed = episode_labels.get(str(r["index"]))
                closure = r.get("closure", r.get("success"))
                if type(closure) is not bool or reviewed not in ("done", "not_done"):
                    n_skip += 1
                    continue
                step = _step(r)
                if not catalog.allows(step):
                    raise ValueError(f"Untrained memory contract: {step.key()}")
                vector = validate_vector(r["vtt"])
                predicates = {"vtt_closure": closure, "reviewed_done": reviewed == "done"}
                scope = _scope(step.key(), args, versions)
                record_id = memory.record(scope, predicates=predicates, terminal=True, episode_id=f"{run.name}/ep{r['episode']}",
                                          source=source + ";closure_labels.json", verifier_kind=args.verifier,
                                          observation_id=f"{run.name}/ep{r['episode']}/step{r['end_step']}")
                store.write(json.dumps({"record_id": record_id, "scope": scope.key(), "skill": step.key(), "vtt": vector.tolist(),
                                        "source": source, "outcome": "success" if all(predicates.values()) else "failure"}) + "\n")
                seen.add(source)
                n_ok += 1
    finally:
        memory.close()
    result = {"promoted": n_ok, "skipped_unverified": n_skip, "db": args.db, "store": str(vtt_store)}
    print(json.dumps(result))
    return result


def export(args):
    memory = Memory(args.db, split_id=args.split, run_id=args.run, readonly=True)
    vectors, record_ids, seen = {}, {}, set()
    context = None
    try:
        for store in map(Path, args.stores):
            for line in store.read_text().splitlines():
                r = json.loads(line)
                if r["outcome"] != "success" or r["record_id"] in seen:
                    continue
                scope = Scope(**json.loads(r["scope"]))
                scope_key = memory._scope_key(scope)
                # Check the actual ledger outcome/provenance, not just a claimed success in the vector file.
                row = memory.connection.execute(
                    "SELECT * FROM evidence WHERE id=? AND scope=? AND verifier_kind IN ('perception_predicates','controller_postconditions')",
                    (r["record_id"], scope_key)).fetchone()
                if (row is None or row["outcome"] not in ("success", "recovery")
                        or json.loads(row["predicates"]).get("reviewed_done") is not True
                        or json.loads(row["predicates"]).get("vtt_closure") is not True):
                    continue
                row_context = asdict(scope)
                row_context.pop("skill_id")
                validate_context(row_context, row_context if context is None else context)
                context = row_context
                if r["skill"] != scope.skill_id or "(" not in scope.skill_id or not scope.skill_id.endswith(")"):
                    raise ValueError("Memory requires an exact skill/object contract, not a class-only key")
                vectors.setdefault(scope.skill_id, []).append(validate_vector(r["vtt"]))
                record_ids.setdefault(scope.skill_id, []).append(r["record_id"])
                seen.add(r["record_id"])
    finally:
        memory.close()
    if context is None:
        raise ValueError("No independently verified successful VTTs to export")
    conditions = {k: validate_vector(np.mean(v, axis=0)).tolist() for k, v in vectors.items()}
    result = {"format": "verified-vtt-v1", "context": context,
              "source": {"db": args.db, "stores": args.stores},
              "counts": {k: len(v) for k, v in vectors.items()}, "conditions": conditions,
              # Support transitions do not count as successful executions conditioned on their mean.
              "arms": {k: [{"arm": "memory0", "source": "ledger:" + ",".join(map(str, record_ids[k])),
                            "successes": 0, "failures": 0, "support_count": len(vectors[k]), "vector": v}] for k, v in conditions.items()}}
    Path(args.out).write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({"exported_contracts": result["counts"], "out": args.out}))
    return result


def _self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run_x"; run.mkdir()
        rows = [{"episode": 1, "index": 1, "skill": "open lid", "objects": ["car"], "closure": True, "skill_steps": 400, "end_step": 900, "vtt": [1.0] * 1024},
                {"episode": 1, "index": 3, "skill": "open door", "objects": ["door"], "closure": False, "skill_steps": 900, "end_step": 1800, "vtt": [0.5] * 1024},
                {"episode": 1, "index": 5, "skill": "pick up from", "objects": ["toy_box", "floors"], "closure": None, "skill_steps": 10, "end_step": 1810, "vtt": [0.1] * 1024},
                {"episode": 1, "index": 6, "key": "close lid(car)", "success": True, "end_step": 1820, "vtt": [0.2] * 1024},
                {"episode": 1, "index": 7, "key": "pick up from(toy_box,floors)", "success": False, "end_step": 1830, "vtt": [0.3] * 1024},
                {"episode": 1, "index": 7, "key": "pick up from(toy_box,floors)", "success": True, "end_step": 1840, "vtt": [0.4] * 1024}]
        (run / "realized_vtt.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (run / "closure_labels.json").write_text(json.dumps({"1": {"1": "done", "3": "not_done", "5": "done", "7": "done"}}))
        db = str(Path(d) / "dev.sqlite")
        a = argparse.Namespace(run_dir=str(run), db=db, split="development", run="run-x", verifier="perception_predicates",
                               scene_context='{"task": "loading_the_car", "scene": "house_double_floor_lower"}', versions=None,
                               wam_version="stage1_skillproto@test", task="task-013")
        assert promote(a)["promoted"] == 2
        assert promote(a)["promoted"] == 0  # repeated promotion is idempotent
        b = argparse.Namespace(db=db, split="development", run="run-x", stores=[str(run / "memory_vtt_store.jsonl")], out=str(Path(d) / "mem.json"))
        export(b)
        m = json.loads((Path(d) / "mem.json").read_text())
        assert list(m["conditions"]) == ["open lid(car)"] and m["counts"] == {"open lid(car)": 1}, m["counts"]
        assert len(m["conditions"]["open lid(car)"]) == 1024 and m["conditions"]["open lid(car)"][0] == 1.0
        # A label without an end step cannot certify multiple retries; an explicit label can.
        (run / "closure_labels.json").write_text(json.dumps({"1": {"7@1840": "done"}}))
        assert promote(a)["promoted"] == 1
        assert export(b)["counts"]["pick up from(toy_box,floors)"] == 1
        # A claimed success in the vector sidecar cannot override a ledger failure.
        sidecar = Path(b.stores[0])
        records = [json.loads(line) for line in sidecar.read_text().splitlines()]
        for record in records:
            record["outcome"] = "success"
        sidecar.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        assert "open door(door)" not in export(b)["conditions"]
        validate_context(m["context"], m["context"])
        try:
            validate_context(m["context"], {**m["context"], "wam_version": "different"})
        except ValueError:
            pass
        else:
            raise AssertionError("Cross-model memory was accepted")
        for vector in ([1.0], [float("nan")] * 1024, [0.0] * 1024):
            try:
                validate_vector(vector)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid VTT was accepted")
        other = Path(d) / "other_scene"
        other.mkdir()
        (other / "realized_vtt.jsonl").write_text(json.dumps(rows[0]) + "\n")
        (other / "closure_labels.json").write_text('{"1":{"1":"done"}}')
        a.run_dir, a.scene_context = str(other), '{"scene":"different"}'
        promote(a)
        b.stores.append(str(other / "memory_vtt_store.jsonl"))
        prior_export = Path(b.out).read_text()
        try:
            export(b)
        except ValueError:
            pass
        else:
            raise AssertionError("Different scene contexts were merged")
        assert Path(b.out).read_text() == prior_export
    print("PASS (synthetic): reviewed exact-contract memory, idempotent promotion, retry-specific evidence, "
          "failure/unknown exclusion, context and vector validation")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote"); p.add_argument("run_dir"); p.add_argument("--db", required=True); p.add_argument("--split", required=True)
    p.add_argument("--run", required=True); p.add_argument("--verifier", default="perception_predicates")
    p.add_argument("--scene-context", required=True); p.add_argument("--versions", default=None)
    p.add_argument("--wam-version", default="stage1_skillproto"); p.add_argument("--task", default="task-013")
    e = sub.add_parser("export"); e.add_argument("--db", required=True); e.add_argument("--split", required=True); e.add_argument("--run", required=True)
    e.add_argument("--stores", nargs="+", required=True); e.add_argument("--out", required=True)
    t = sub.add_parser("self-test")
    args = ap.parse_args()
    {"promote": promote, "export": export, "self-test": lambda a: _self_test()}[args.cmd](args)


if __name__ == "__main__":
    main()
