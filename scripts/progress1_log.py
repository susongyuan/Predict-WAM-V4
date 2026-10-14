#!/usr/bin/env python3
"""Append a short, locked pre-operation record for the task-1/2 workflow."""
import argparse
import fcntl
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]

def refresh_active_run(text):
    """Refresh only this project's explicitly linked run; never count probes."""
    import json,re
    match=re.search(r'^<!-- active_zhuhai_pipeline: (.+) -->$',text,re.M)
    if not match:return text
    root=(ROOT/match.group(1).strip()).resolve()
    if not root.is_relative_to((ROOT/'outputs').resolve()):return text
    def count(task,mode):
        path=root/task/mode/'manifest.json'
        if not path.exists():return 0
        try:
            value=json.loads(path.read_text())
            if value.get('contract',{}).get('source_probe',False):return 0
            number=value.get('success_count',0)
            return number if type(number) is int and number>=0 else '?'
        except (OSError,ValueError):return '?'
    line=('- **生产数据：task1源'+str(count('zhuhai_task1','source'))+'/100、随机化'+str(count('zhuhai_task1','randomized'))
          +'/500；task2源'+str(count('zhuhai_task2','source'))+'/100、随机化'+str(count('zhuhai_task2','randomized'))+'/500。**')
    text=re.sub(r'^- \*\*生产数据：.*?\*\*$',lambda _:line,text,flags=re.M)
    try:
        state=json.loads((root/'status.json').read_text())
        stage=state.get('active_stage') or '无活动阶段'
        status_line=f"- **活动流水线：{state.get('status','unknown')} / {stage}**；以run内清单和评分为准。"
        text=re.sub(r'^- \*\*活动流水线：.*$',lambda _:status_line,text,flags=re.M)
    except (OSError,ValueError):pass
    return text

def log(message, actor="root"):
    """Archive every event; publish at most one compact Markdown summary per 600s."""
    import json, re
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
        published = [line for line in history.splitlines() if " — 10分钟汇总：" in line]
        times = [datetime.fromisoformat(line.split(" — ", 1)[0].removeprefix("- ")) for line in published]
        last = max(times) if times else None
        if last is not None and (stamp - last).total_seconds() < 600:
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
        published.append(f"- {record['time']} — 10分钟汇总：{summary}")
        # ponytail: curated milestones stay intact; full per-event history stays in JSONL.
        if len(published) > 24:
            older, published = published[:-24], published[-24:]
            compressed = (f"- 滚动记录压缩摘要：较早{len(older)}次定时记录，起点[{older[0]}]，"
                          f"末态[{older[-1]}]；中间完整流水见docs/progress_archive/progress1_operations.jsonl。\n")
            # Keep one bounded rolling-summary entry instead of adding a new
            # prefix line every ten minutes forever. Original Markdown bytes
            # and the full per-operation journal remain recoverable.
            import zipfile
            archive_zip=ROOT/'docs/archive/documentation_snapshot_20261007.zip'
            archive_zip.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(archive_zip,'a',compression=zipfile.ZIP_DEFLATED) as zipped:
                zipped.writestr('progress1_rolling/'+stamp.strftime('%Y%m%dT%H%M%S%f')+'.md',old)
            prefix=re.sub(r'^- 滚动记录压缩摘要：.*\n?', '', prefix, flags=re.M)
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
