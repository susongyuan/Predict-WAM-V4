#!/usr/bin/env python3
"""v41 successor of progress1_log: same locked journal, 1800s published summaries.

The user's 2026-10-12 requirement raises the published summary cadence from
600s to 1800s and says the new summary must cover the historical 10-minute
summaries. The frozen scripts/progress1_log.py is left untouched for the
already-recorded runs. Every event still lands in the JSONL immediately;
progress1.md still receives at most one compact summary line per interval.
"""
import argparse
import fcntl
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SECONDS = 1800
SUMMARY_LABEL = "30分钟汇总"


def log(message, actor="root"):
    """Archive every event; publish at most one compact Markdown summary per 1800s."""
    import json, re, zipfile
    from collections import defaultdict
    stamp = datetime.now(ZoneInfo("Asia/Shanghai"))
    record = dict(time=stamp.isoformat(timespec="seconds"), actor=actor, message=" ".join(message.split()))
    archive = ROOT / "docs/progress_archive"
    archive.mkdir(parents=True, exist_ok=True)
    journal = archive / "progress1_operations.jsonl"
    marker = "## 关键操作记录\n"
    with (ROOT / "progress1.md").open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        old = handle.read()
        if marker not in old:
            raise ValueError("progress1 must retain its key-record section; do not append an unbounded log")
        prefix, history = old.split(marker, 1)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        published = [line for line in history.splitlines()
                     if " — 10分钟汇总：" in line or " — 30分钟汇总：" in line]
        times = [datetime.fromisoformat(line.split(" — ", 1)[0].removeprefix("- ")) for line in published]
        last = max(times) if times else None
        if last is not None and (stamp - last).total_seconds() < SUMMARY_SECONDS:
            return False
        groups = defaultdict(list)
        for raw in journal.read_text(encoding="utf-8").splitlines():
            item = json.loads(raw)
            when = datetime.fromisoformat(item["time"])
            if last is None or when > last:
                groups[item.get("actor", "unknown")].append(item["message"])
        pieces = []
        for who, messages in groups.items():
            unique = list(dict.fromkeys(" ".join(m.split()) for m in messages))
            important = [m for m in unique[:-1] if re.search(r"失败|异常|中断|停止|授权|变更", m)]
            detail = "；".join(list(dict.fromkeys(important + unique[-1:])))
            pieces.append(f"{who}（{len(messages)}次）：{detail}")
        summary = "；".join(pieces) or record["message"]
        published.append(f"- {record['time']} — {SUMMARY_LABEL}：{summary}")
        # Curated milestones stay intact; full per-event history stays in JSONL.
        if len(published) > 24:
            older, published = published[:-24], published[-24:]
            compressed = (f"- 滚动记录压缩摘要：较早{len(older)}次定时记录，起点[{older[0]}]，"
                          f"末态[{older[-1]}]；中间完整流水见docs/progress_archive/progress1_operations.jsonl。\n")
            archive_zip = ROOT / 'docs/archive/documentation_snapshot_20261007.zip'
            archive_zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_zip, 'a', compression=zipfile.ZIP_DEFLATED) as zipped:
                zipped.writestr('progress1_rolling/' + stamp.strftime('%Y%m%dT%H%M%S%f') + '.md', old)
            prefix = re.sub(r'^- 滚动记录压缩摘要：.*\n?', '', prefix, flags=re.M)
            prefix += compressed
        content = prefix + marker + "\n" + "\n".join(published) + "\n"
        if len(content.splitlines()) >= 200:
            raise ValueError("Curated summaries need consolidation before 200 lines; complete event already archived")
        handle.seek(0); handle.write(content); handle.truncate(); handle.flush()
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--actor", default="root")
    args = parser.parse_args()
    log(args.message, args.actor)
