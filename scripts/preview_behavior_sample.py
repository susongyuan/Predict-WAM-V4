#!/usr/bin/env python3
"""Decode six timestamp-checked demonstration frames; no simulator or GPU.

Run using .conda-envs/behavior/bin/python after its PyAV dependency is installed.
This saves an offline sample, not a closed-loop evaluation result.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.behavior_data import BehaviorData, decode_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int)
    parser.add_argument('--anchor', type=int, default=0)
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--output', type=Path, default=Path('outputs/data_preview'))
    args = parser.parse_args()
    from PIL import Image, ImageDraw
    import numpy as np
    data = BehaviorData(split=args.split)
    episode = args.episode if args.episode is not None else data.episode_ids[0]
    sample = data.sample(episode, args.anchor)
    destination = args.output / f'episode_{episode}_anchor_{args.anchor}'
    destination.mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (960, 760), '#ffffff')
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), 'OFFLINE DEMONSTRATION / not a simulator rollout', fill='#263544')
    manifest = {'episode': episode, 'anchor': args.anchor, 'split': args.split,
                'status': 'offline_frame_alignment_only', 'frames': []}
    for row, key in enumerate(('frames', 'future_frames')):
        for column, (camera, reference) in enumerate(sample[key].items()):
            rgb, actual = decode_reference(reference, data.fps)
            name = f'{key}_{camera.rsplit(".", 1)[-1]}.png'
            rgb.save(destination / name)
            manifest['frames'].append(dict(reference, group=key, actual_pts_s=actual,
                                           error_s=actual-reference['timestamp_s'], file=name))
            rgb.thumbnail((300, 300))
            x, y = column * 320 + 10, row * 350 + 55
            draw.text((x, y), f'{camera.split(".")[-1]} | t+{row * 32}', fill='#263544')
            sheet.paste(rgb, (x, y+24))
    np.savez_compressed(destination / 'lowdim.npz', state=sample['state'],
                        action_sequence=sample['action_sequence'])
    sheet.save(destination / 'contact_sheet.png')
    (destination / 'alignment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'decoded_frames': len(manifest['frames']),
                      'max_pts_error_s': max(abs(f['error_s']) for f in manifest['frames']),
                      'output': str(destination)}))


if __name__ == '__main__':
    main()
