#!/usr/bin/env python3
"""Inspect a running policy's question or answer its exact episode/question IDs.

Run with .conda-envs/lilawam/bin/python. Choice indices are zero based.
    scripts/clarify_policy.py
    scripts/clarify_policy.py --episode 1 --question-id 1 --answer 0
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from websockets.sync.client import connect
from harness.policy_server import Packer, unpackb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--question-id", type=int)
    parser.add_argument("--answer", type=int)
    args = parser.parse_args()
    provided = [args.episode is not None, args.question_id is not None, args.answer is not None]
    if any(provided) and not all(provided):
        parser.error("An answer requires --episode, --question-id and --answer")
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    msg = {"status": True} if args.answer is None else {"clarification": {
        "episode": args.episode, "question_id": args.question_id, "answer": args.answer}}
    with connect(f"ws://127.0.0.1:{args.port}", open_timeout=10, max_size=None) as ws:
        unpackb(ws.recv(timeout=30))  # The official protocol sends metadata first.
        ws.send(Packer().pack(msg))
        result = unpackb(ws.recv(timeout=30))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("accepted") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
