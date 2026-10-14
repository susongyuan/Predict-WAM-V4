#!/usr/bin/env python3
"""Render a user-requested video from immutable onboard archives, without simulation."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/zhuhai2026_tasks12"
SUMMARY = BASE / "wam_precision_v2_task2_dagger_70k_fixed5_20261011_v23/summary.json"
OUT = ROOT / "vieo/70k_seed932101_close_without_grasp_40s_70s.mp4"
START, END, FPS = 400, 700, 10


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    assert not OUT.exists(), "Preserve the existing requested video"
    OUT.parent.mkdir(exist_ok=True)
    cohort = json.loads(SUMMARY.read_text())
    row = next(r for r in cohort["episodes"] if r["seed"] == 932101)
    directory = Path(row["raw_result_path"]).parent
    assert sha(directory / "result.json") == row["raw_result_sha256"]
    assert row["full_task_success"] is False and row["observed_base_score"] == 0
    evidence = row["evidence_files"]
    for name in ("actions.jsonl", "rule_evidence.jsonl", "onboard/frames.jsonl"):
        assert sha(directory / name) == evidence[name]["sha256"]
    rules = {}
    with (directory / "rule_evidence.jsonl").open() as stream:
        for line in stream:
            rule = json.loads(line)
            if START <= rule["control_step"] < END:
                items = rule["score"]["items"]
                rules[rule["control_step"]] = dict(
                    time=rule["physics_time_seconds"], score=rule["score"]["base_score"],
                    contact=any(v["touching_gripper"] for v in rule["objects"].values()),
                    grasp=any(v["grasp_awarded"] for v in items.values()))
    assert len(rules) == END - START
    ffmpeg = next((ROOT / ".conda-envs/behavior6/lib/python3.12/site-packages/imageio_ffmpeg/binaries").glob("ffmpeg-linux-*"))
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title_font = ImageFont.truetype(font_path, 30)
    font = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 22)
    command = [str(ffmpeg), "-nostdin", "-n", "-loglevel", "warning", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-video_size", "1440x900", "-framerate", str(FPS),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)]
    frames = []
    log_path = OUT.with_suffix(".encoder.log")
    with log_path.open("xb") as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for step in range(START, END):
                name = f"onboard/observation_{step:06d}.npz"
                path = directory / name
                assert sha(path) == evidence[name]["sha256"], "Recorded observation changed"
                with np.load(path, allow_pickle=False) as archive:
                    measured = archive["measured_joint_gripper_state"].copy()
                    commanded = archive["commanded_joint_gripper_state"].copy()
                    images = {name: Image.fromarray(archive[name]) for name in
                              ("head_camera", "left_camera", "right_camera")}
                canvas = Image.new("RGB", (1440, 900), "#101820")
                draw = ImageDraw.Draw(canvas)
                r = rules[step]
                draw.text((20, 8), "70k 失败实例｜seed 932101｜原始测试观测回放（1×）", font=title_font, fill="white")
                draw.text((20, 50), f"任务时间 {r['time']:.1f} 秒｜左夹爪：目标 {commanded[7]:.2f} / 实际 {measured[7]:.2f}（1开，0合）", font=font, fill="#bdd6ee")
                canvas.paste(images["head_camera"].resize((960, 720), Image.Resampling.LANCZOS), (0, 100))
                canvas.paste(images["left_camera"].resize((480, 360), Image.Resampling.LANCZOS), (960, 100))
                canvas.paste(images["right_camera"].resize((480, 360), Image.Resampling.LANCZOS), (960, 460))
                for xy, label in [((12, 105), "头部相机：全景"), ((972, 105), "左腕相机"), ((972, 465), "右腕相机")]:
                    box = draw.textbbox(xy, label, font=small)
                    draw.rectangle((box[0]-5, box[1]-3, box[2]+5, box[3]+3), fill="#101820")
                    draw.text(xy, label, font=small, fill="white")
                contact = "有" if r["contact"] else "无"
                grasp = "有" if r["grasp"] else "无"
                draw.text((20, 833), f"物体接触：{contact}｜有效抓持：{grasp}｜得分：{r['score']}    关注约53秒：合拢夹爪仍未抓到物体", font=font, fill="#ffca82")
                draw.text((20, 868), "由已存的三路机载观测生成；没有重跑测试，也没有添加外部相机。", font=small, fill="#d6dce4")
                encoder.stdin.write(canvas.tobytes())
                frames.append(dict(control_step=step, physics_time_seconds=r["time"],
                                   observation_sha256=evidence[name]["sha256"]))
                if step == 540:
                    canvas.save(OUT.with_suffix(".preview.png"))
        finally:
            encoder.stdin.close()
            code = encoder.wait(timeout=60)
    assert code == 0, "Video encoder failed; preserve its log"
    decoder = cv2.VideoCapture(str(OUT))
    assert decoder.isOpened(), "Video cannot be decoded"
    decoded, dimensions = 0, None
    while True:
        ok, frame = decoder.read()
        if not ok:
            break
        decoded += 1
        dimensions = [frame.shape[1], frame.shape[0]]
    fps = decoder.get(cv2.CAP_PROP_FPS)
    decoder.release()
    assert decoded == END - START and fps == FPS and dimensions == [1440, 900]
    metadata = dict(schema="user-requested-archived-WAM-failure-video-v23",
                    time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                    authorization="User explicitly requested one classic failure video saved in vieo",
                    selection="First fixed70k seed, first left close; predeclared selected episode remains score0",
                    milestone=70000, seed=932101, full_task_success=False, score=0,
                    physical_seconds=[START/FPS, END/FPS], playback_speed=1, fps=FPS,
                    decoded_frames=decoded, dimensions=dimensions, video_path=str(OUT),
                    video_sha256=sha(OUT), video_size=OUT.stat().st_size,
                    source_summary=dict(path=str(SUMMARY), sha256=sha(SUMMARY)),
                    source_result=dict(path=row["raw_result_path"], sha256=row["raw_result_sha256"]),
                    source_action_sha256=evidence["actions.jsonl"]["sha256"],
                    source_rule_sha256=evidence["rule_evidence.jsonl"]["sha256"],
                    source_observations=frames,
                    rendering="H264 annotated montage of existing head+left-wrist+right-wrist RGB; enlarged from320x240, no new detail claimed",
                    evaluation_protocol_unchanged=True, new_simulator_attempts=0, external_cameras_added=0,
                    commanded_values="NPZ commanded state: previously dispatched interval, not a future model command",
                    renderer_sha256=sha(__file__), encoder_sha256=sha(ffmpeg))
    with OUT.with_suffix(".provenance.json").open("x") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(video=str(OUT), duration_seconds=decoded/fps,
                         size=OUT.stat().st_size, decode_verified=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
