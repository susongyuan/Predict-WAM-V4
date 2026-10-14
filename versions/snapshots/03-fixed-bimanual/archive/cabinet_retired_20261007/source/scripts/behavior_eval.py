#!/usr/bin/env python3
"""Build or run the pinned official policy eval command under conservative resource monitoring."""
import argparse
import json
import shlex
import sys
from behavior_preflight import CONFIG, ENV, ROOT, SOURCE, child_environment, collect

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.task_profile import TASK_ID, TASK_NAME  # noqa: E402  P2_TASK_ID; default 13 = CONFIG task loading_the_car


def command(profile, host, port):
    if not 1 <= port <= 65535:
        raise ValueError("port must be in 1..65535")
    args = [str(ENV / "bin/python"), "-m", "omnigibson.eval.eval", "--task-name", TASK_NAME,
            "--robot-config", str(SOURCE / "OmniGibson/omnigibson/eval/r1pro.yaml"),
            "--policy", "websocket", "--mode", "public_test", "--instance-indices",
            *map(str, CONFIG["profiles"][profile]), "--num-rollouts", "1", "--headless",
            "--host", host, "--port", str(port), "--output-dir", str(ROOT / "outputs/behavior" / profile)]
    if profile == "smoke":
        args.extend(["--max-steps", "10"])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=CONFIG["profiles"], default="smoke")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wall-timeout", type=int, default=None, help="Optional process wall timeout; aborted runs are not valid scores")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if TASK_ID == 13:  # the default task keeps the configured official task name
            assert TASK_NAME == CONFIG["task"]["name"] == "loading_the_car"
        assert command("smoke", "127.0.0.1", 8000)[3:5] == ["--task-name", TASK_NAME]
        assert "--max-steps" in command("smoke", "127.0.0.1", 8000)
        for profile in ("development", "report"):
            assert "--max-steps" not in command(profile, "127.0.0.1", 8000)
            assert "local" not in command(profile, "127.0.0.1", 8000)
        print("PASS: official timeouts retained; websocket policy required; no all-zero local action")
        return
    env = child_environment()
    argv = command(args.profile, args.host, args.port)
    print(json.dumps({"profile": args.profile, "executed": False, "environment": env, "argv": argv,
                      "shell_command_for_review": shlex.join(["env", *[f"{k}={v}" for k, v in env.items()], *argv]),
                      "note": "Policy eval requires a live websocket policy. Use behavior_scene_smoke.py for reset/sensors without a policy. Monitoring cannot guarantee instantaneous GPU utilization or VRAM caps."},
                     ensure_ascii=False, indent=2))
    if args.execute:
        blockers = collect()["blockers"]
        if blockers:
            print(json.dumps({"blocked": blockers}, ensure_ascii=False, indent=2))
            raise SystemExit(2)
        from behavior_resources import run_guarded
        raise SystemExit(run_guarded(argv, env, ROOT / "outputs/behavior" / args.profile,
                                     timeout=args.wall_timeout))


if __name__ == "__main__":
    main()
