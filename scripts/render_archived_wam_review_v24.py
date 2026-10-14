#!/usr/bin/env python3
"""User-requested annotated replay of recorded onboard images; never re-runs physics."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def render(args):
    summary = Path(args.summary).resolve()
    out = Path(args.out).resolve()
    assert out.parent == ROOT / "success video", "Use the user's existing video directory"
    assert not out.exists(), "Preserve recorded artifacts"
    cohort = json.loads(summary.read_text())
    row = next(r for r in cohort["episodes"] if r["seed"] == args.seed)
    directory = Path(row["raw_result_path"]).parent
    assert sha(directory / "result.json") == row["raw_result_sha256"]
    evidence = row["evidence_files"]
    assert sha(directory / "rule_evidence.jsonl") == evidence["rule_evidence.jsonl"]["sha256"]
    start, end, fps = round(args.start * 10), round(args.end * 10), 10
    rules = {}
    with (directory / "rule_evidence.jsonl").open() as stream:
        for line in stream:
            rule = json.loads(line)
            step = rule["control_step"]
            if start <= step < end:
                item = rule["score"]["items"]["bowl"]
                rules[step] = dict(time=rule["physics_time_seconds"], score=rule["score"]["base_score"],
                    contact=rule["objects"]["bowl"]["touching_gripper"],
                    grasp=item["grasp_awarded"], placed=item["placement_valid"], hold=item["hold_seconds"])
    assert len(rules) == end - start
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    large = ImageFont.truetype(font_path, 30)
    font = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 22)
    ffmpeg = next((ROOT / ".conda-envs/behavior6/lib/python3.12/site-packages/imageio_ffmpeg/binaries").glob("ffmpeg-linux-*"))
    command = [str(ffmpeg), "-nostdin", "-n", "-loglevel", "warning", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-video_size", "1440x900", "-framerate", "10", "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out)]
    provenance = []
    with out.with_suffix(".encoder.log").open("xb") as log:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for step in range(start, end):
                name = f"onboard/observation_{step:06d}.npz"
                path = directory / name
                assert sha(path) == evidence[name]["sha256"], "Recorded source observation changed"
                with np.load(path, allow_pickle=False) as z:
                    q = z["measured_joint_gripper_state"].copy()
                    u = z["commanded_joint_gripper_state"].copy()
                    images = [Image.fromarray(z[k]) for k in ("head_camera", "left_camera", "right_camera")]
                r = rules[step]
                canvas = Image.new("RGB", (1440, 900), "#101820")
                draw = ImageDraw.Draw(canvas)
                draw.text((20, 8), f"{args.label}｜seed {args.seed}｜原始观测原速回放", font=large, fill="white")
                draw.text((20, 50), f"任务时间 {r['time']:.1f} 秒｜左夹爪：目标 {u[7]:.2f} / 实际 {q[7]:.2f}（1开，0合）", font=font, fill="#bdd6ee")
                for image, size, origin in zip(images, ((960,720),(480,360),(480,360)), ((0,100),(960,100),(960,460))):
                    canvas.paste(image.resize(size, Image.Resampling.LANCZOS), origin)
                for xy, label in (((12,105),"头部相机：全景"),((972,105),"左腕相机"),((972,465),"右腕相机")):
                    box = draw.textbbox(xy, label, font=small)
                    draw.rectangle((box[0]-5,box[1]-3,box[2]+5,box[3]+3),fill="#101820")
                    draw.text(xy,label,font=small,fill="white")
                flag = lambda x: "有" if x else "无"
                draw.text((20,833), f"碗接触：{flag(r['contact'])}｜有效抓持：{flag(r['grasp'])}｜有效放置：{flag(r['placed'])}｜得分 {r['score']}", font=font, fill="#ffca82")
                draw.text((20,868), "局部抓碗成功与整任务成功分开；三路机载记录，未重跑测试。", font=small, fill="#d6dce4")
                child.stdin.write(canvas.tobytes())
                provenance.append(dict(control_step=step, physics_time_seconds=r["time"], observation_sha256=evidence[name]["sha256"]))
                if step == start + min(140, end - start - 1):
                    canvas.save(out.with_suffix(".preview.png"))
        finally:
            child.stdin.close()
            code = child.wait(timeout=60)
    assert code == 0, "Encoder failed; preserve its log"
    decoder = cv2.VideoCapture(str(out))
    assert decoder.isOpened()
    count = 0
    while True:
        ok, frame = decoder.read()
        if not ok:
            break
        assert frame.shape[:2] == (900,1440)
        count += 1
    measured_fps = decoder.get(cv2.CAP_PROP_FPS)
    decoder.release()
    assert count == end - start and measured_fps == fps
    metadata = dict(schema="user-requested-30k-success-failure-onboard-replays-v24",
        time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        milestone=args.milestone, seed=args.seed, label=args.label, duration_seconds=count/fps,
        selection_seconds=[args.start,args.end], playback_speed=1, fps=fps, decoded_frames=count,
        source_summary=dict(path=str(summary),sha256=sha(summary)),
        source_result=dict(path=row["raw_result_path"],sha256=row["raw_result_sha256"]),
        video=dict(path=str(out),sha256=sha(out),size=out.stat().st_size),
        source_observations=provenance, renderer_sha256=sha(__file__), arguments=vars(args),
        whole_task_success=row["full_task_success"], final_score=row["observed_base_score"],
        observation_resolution=[320,240], montage_resolution=[1440,900],
        simulator_attempts_added=0, external_cameras_added=0, evaluation_protocol_unchanged=True)
    with out.with_suffix(".provenance.json").open("x") as stream:
        json.dump(metadata,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(video=str(out),seconds=count/fps,frames=count,decode_verified=True),ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--milestone",type=int,required=True)
    p.add_argument("--start",type=float,required=True)
    p.add_argument("--end",type=float,required=True)
    p.add_argument("--label",required=True)
    p.add_argument("--out",required=True)
    render(p.parse_args())
