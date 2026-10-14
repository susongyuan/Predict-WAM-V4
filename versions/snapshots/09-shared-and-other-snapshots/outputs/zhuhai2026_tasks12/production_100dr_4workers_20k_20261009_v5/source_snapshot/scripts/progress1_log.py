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
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    record = f"- {stamp} — {actor}：{' '.join(message.split())}\n"
    archive = ROOT / "docs/progress_archive"
    archive.mkdir(parents=True, exist_ok=True)
    with (ROOT / "progress1.md").open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        old = refresh_active_run(handle.read())
        with (archive / "progress1_operations.jsonl").open("a", encoding="utf-8") as audit:
            import json
            audit.write(json.dumps(dict(time=stamp, actor=actor, message=message), ensure_ascii=False) + "\n")
        marker = "## 关键操作记录\n"
        if marker in old:
            prefix, history = old.split(marker, 1)
            lines = [line for line in history.splitlines() if line.strip()]
            keep = max(1, 195 - len((prefix + marker).splitlines()))
            content = prefix + marker + "\n" + "\n".join((lines + [record.rstrip()])[-keep:]) + "\n"
        else:
            content = old + record
        handle.seek(0)
        handle.write(content)
        handle.truncate()
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--actor", default="root")
    args = parser.parse_args()
    log(args.message, args.actor)
