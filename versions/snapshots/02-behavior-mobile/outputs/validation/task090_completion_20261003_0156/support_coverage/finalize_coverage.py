#!/usr/bin/env python3
"""Package bounded TRAIN visibility inspection; no video/model/runtime execution."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/user/Robotic/project2')
OUT = ROOT / 'outputs/validation/task090_completion_20261003_0156/support_coverage'
BASE = ROOT / 'outputs/validation/task090_completion_20261003_0156'
OLD = ROOT / 'outputs/validation/task090_onboard_grasp_20261002'
DATA = ROOT / 'data/behavior-1k/2026-challenge-demos'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    cached = load(OUT / 'cache_manifest.json')
    native = load(OUT / 'native_manifest.json')
    states = load(OUT / 'states.json')
    prenav = load(OLD / 'prenav_views_v1/manifest.json')
    wrist = load(OLD / 'wrist_views_v1/manifest.json')
    skill_file = DATA / 'skill_segments_task090.json'
    skills = load(skill_file)
    for ep, end in [(18000, 480), (18001, 870)]:
        spec = skills['episodes'][str(ep)]
        assert spec['split'] == 'train'
        seg = spec['segments'][0]
        assert (seg['start'], seg['end'], seg['skill_type']) == (0, end, 'navigation')
    bank = {(s['episode'], s['frame'], s['camera_label']): s for s in cached['samples']}
    verified_images = set()
    for row in cached['samples']:
        assert digest(row['image']) == row['sha256']
        assert digest(row['cache_metadata']) == row['cache_metadata_sha256']
        assert digest(row['cache_index']) == row['cache_index_sha256']
        idx = np.load(row['cache_index'], mmap_mode='r')
        i = row['cache_index_row']
        start, end = int(idx[i]), int(idx[i + 1])
        assert [start, end] == row['cache_byte_interval']
        with Path(row['cache_pack']).open('rb') as f:
            f.seek(start)
            assert hashlib.sha256(f.read(end - start)).hexdigest() == row['sha256']
        assert Path(row['rgb_video_source']).is_file()
        verified_images.add(row['image'])
    for row in states:
        assert digest(row['source']) == row['source_sha256']
        arr = np.load(row['source'], mmap_mode='r')
        assert np.array_equal(arr[row['row_index']], np.array(row['state61'], dtype=arr.dtype))
    for row in prenav['samples']:
        assert digest(row['rgb']) == row['sha256']
        verified_images.add(row['rgb'])
    for row in native['samples']:
        assert digest(row['image']) == row['sha256']
        assert Path(row['source']).stat().st_size == row['source_bytes']
        assert row['actual_video_pts_s'] == row['requested_video_pts_s']
        verified_images.add(row['image'])
    continuous = BASE / 'continuous_train/extracted/frames.jsonl'
    exact = {}
    with continuous.open() as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            key = (row['episode'], row['frame_index'])
            if key not in [(18000, 480), (18001, 870)]:
                continue
            assert digest(row['rgb']) == row['rgb_sha256']
            assert digest(row['depth_m']) == row['depth_m_sha256']
            verified_images.add(row['rgb'])
            exact[key] = {
                'manifest': str(continuous), 'line_1based': line_number,
                'episode': row['episode'], 'frame': row['frame_index'],
                'camera': 'observation.rgb.zed_link_camera_0',
                'image': row['rgb'], 'image_sha256': row['rgb_sha256'],
                'actual_video_pts_s': row['rgb_decoded_pts_s'],
                'depth': row['depth_m'], 'depth_sha256': row['depth_m_sha256'],
                'depth_actual_video_pts_s': row['depth_decoded_pts_s'],
                'depth_unit': 'm', 'depth_pairing': 'pre-existing exact same-PTS cache; hash verified, not newly decoded',
                'state_source': row['lowdim_source'], 'dataset_row_index': row['dataset_index'],
                'state61_and_camera_pose7': 'in referenced manifest line; no physical outcome interpretation',
            }
    for row in wrist['samples']:
        if row['frame'] == 870:
            assert digest(row['rgb']) == row['sha256']
            verified_images.add(row['rgb'])
            wrist_reference = {'manifest': str(OLD / 'wrist_views_v1/manifest.json'), **row,
                               'camera': 'observation.rgb.right_realsense_link_camera_0', 'depth': None}

    best = native['samples'][0]
    assert (best['episode'], best['frame']) == (18001, 724)
    best_state = next(s for s in states if (s['episode'], s['frame']) == (18001, 724))
    best_provenance = {
        'episode': 18001, 'split': 'train', 'frame': 724, 'camera': best['camera'],
        'image': best['image'], 'sha256': best['sha256'], 'image_size': best['size'],
        'source_video': best['source'], 'source_revision': best['source_revision'],
        'episode_time_s': best['episode_time_s'], 'episode_offset_s': best['episode_offset_s'],
        'actual_video_pts_s': best['actual_video_pts_s'], 'actual_pts': best['actual_pts'],
        'time_base': best['time_base'], 'pts_error_s': best['pts_error_s'],
        'state_source': best['state_source'], 'state_row': 724,
        'state_source_sha256': best_state['source_sha256'],
        'state_values': str(OUT / 'states.json'),
        'state_semantics': 'recorded observation.state, 61D; not contact/holding or physical outcome',
        'phase': 'original navigation [0,870), selected for visual coverage only',
        'depth': None, 'depth_note': 'No exact f724 head-depth cache established here; f870 depth must not substitute for f724.',
        'synchronous_wrist_images': [bank[(18001, 724, cam)]['image'] for cam in ['left', 'right']],
        'wrist_pts_provenance': 'Episode offset + frame/30 from each existing cache metadata; not a new video decode.',
    }
    visibility = {
        'source': 'assistant_visual_inspection',
        'vocabulary': {'Yes': 'visually resolved within stated 2D scope only',
                       'Unknown': 'occluded, cropped, absent, undersampled or not a directly observable physical relation'},
        'best_navigation_view': {
            'episode': 18001, 'frame': 724, 'camera': 'head',
            'whole_projected_bin_outline_in_frame': 'Yes',
            'visible_front_lower_edge': 'Yes',
            'entire_bottom_surface_visible': 'Unknown',
            'adjacent_ground_visible': 'Yes',
            'projected_front_bin_ground_boundary_separable': 'Yes',
            'actual_contact_interface_separable': 'Unknown',
            'same_target_overlap_head_left_right': 'Unknown',
            'reason': '头图可分黑桶前侧下缘与浅色地面，物体轮廓未被画幅裁切；底面/背侧与接触内部不可见，阴影不是接触证明。同步腕图主要为机体与周边地面，未见可对应桶区域。',
        },
        'sample_groups': [
            {'episode': 18000, 'frames': [90, 180], 'camera': 'head',
             'bottom': 'Unknown', 'target_adjacent_ground': 'Unknown', 'contact_boundary': 'Unknown',
             'reason': '已有导航原图未找到目标桶；可见房间地面不能绑定成目标支撑。'},
            {'episode': 18000, 'frames': [220, 240, 260, 270, 280, 300, 320], 'camera': 'head',
             'bottom': 'Unknown', 'target_adjacent_ground': 'Yes', 'contact_boundary': 'Unknown',
             'whole_projected_outline': 'Unknown', 'reason': '桶在左侧和/或下沿被裁切；附近地面可见不等于底面可见。'},
            {'episode': 18000, 'frames': [360, 420, 480], 'camera': 'head',
             'bottom': 'Unknown', 'target_adjacent_ground': 'Yes', 'contact_boundary': 'Unknown',
             'whole_projected_outline': 'Unknown', 'reason': '桶底出画幅下沿，包含导航原图360及操作窗起始480。'},
            {'episode': 18001, 'frames': [0, 144, 290, 434, 580], 'camera': 'head',
             'bottom': 'Unknown', 'target_adjacent_ground': 'Unknown', 'contact_boundary': 'Unknown',
             'whole_projected_outline': 'Unknown', 'reason': '目标不在可辨完整视野或被厨房遮挡；地面虽存在，未建立目标局部支撑对应。'},
            {'episode': 18001, 'frames': [840, 870], 'camera': 'head',
             'bottom': 'Unknown', 'target_adjacent_ground': 'Yes', 'contact_boundary': 'Unknown',
             'whole_projected_outline': 'Unknown', 'reason': '桶移至画面右下并裁切；不能用此前724的轮廓替代当前可见性。'},
            {'episodes': [18000, 18001], 'frames': 'all 16 times listed in cache_manifest.json',
             'cameras': ['left', 'right'], 'bottom': 'Unknown', 'target_adjacent_ground': 'Unknown',
             'contact_boundary': 'Unknown', 'same_target_overlap_with_head': 'Unknown',
             'reason': '32张腕部缓存均以黑色机体遮挡为主，仅见周边地面/家具；870右腕已有原图复核相同遮挡。共同看到某些地面不构成桶底区域跨相机重叠。',
             'depth': None},
        ],
        'global_unknowns': ['完整三维外包络（尤其背面、底面）', '支撑足迹及完整接触界面',
                            '真实支撑/离地/抓持/释放关系', '已验证的导航至操作身份连续性', '同目标跨相机覆盖'],
        'coverage_limit': '仅上述稀疏既有样本；没有遍历导航全窗，不声称18000所有未检帧均不可用。',
    }
    next_action = {
        'priority': 1,
        'name': '优先复用导航的头部目标—地面观测，并以仅观察式重构图补齐覆盖',
        'implemented': False,
        'why': '18001/724已提供可复用二维全桶/邻地面视角，但未提供完整底面或物理支撑；18000已检样本无同等覆盖。不能把当前局部点云/原生尺寸包装成完整外包络。',
        'observation_change': '下次必要运行沿既有导航先观察：在进入近距离裁切前暂停靠近并保持头部RGB-D对准桶的全可见轮廓、前下缘和紧邻地面。18001优先复用724对应构图；18000应更早向桶侧偏转头部（当前图左侧），再适当下俯；若仍裁切，仅用导航保持更远观察位置或小侧移找下缘视线。具体角度/距离由实际构图检查决定，不从这些2D图臆定数值。不得移桶、重放上身答案或改WAM控制。',
        'identity_across_navigation_manipulation': [
            '以运行时当前episode/attempt/target ID绑定同步RGB-D、PTS、相机标定/位姿、state与支撑候选区域；历史TRAIN帧仅作构图证据，不注入新回合身份。',
            '导航期间持续维护同一目标ID与不可变已观测3D锚；在连续接受的短距外观观测之间交接，保留原始证据、时间及误差，不在切入操作窗口时重置或重新绑定近手候选。',
            '跨相机仅在真实同目标重叠且标定/时间一致时交接；现有腕RGB不可冒充深度。失去可见性/对应则关系保持Unknown，通过仅观察重获，而不是从阶段、闭爪或旧帧推断抓持。',
        ],
        'stop_condition': '即使前下缘与地面可分，底面/接触仍不可分时继续Unknown；本动作不保证完整外包络可观测，也不构成新物理成功或准入。',
    }
    result = {
        'status': 'visibility_review_complete_not_physical_admission',
        'source': 'assistant_visual_inspection',
        'best_navigation_candidate': best_provenance,
        'inspection': visibility,
        'initial_exact_rgbd_references': list(exact.values()),
        'wrist_native_confirmation': wrist_reference,
        'provenance': {
            'cache_manifest': str(OUT / 'cache_manifest.json'), 'native_manifest': str(OUT / 'native_manifest.json'),
            'state_rows': str(OUT / 'states.json'), 'prior_navigation_manifest': str(OLD / 'prenav_views_v1/manifest.json'),
            'original_skill_windows': {'source': str(skill_file), 'sha256': digest(skill_file),
                                      '18000': [0, 480], '18001': [0, 870], 'interval': 'half-open',
                                      'use': 'window selection only, not outcome labels'},
            'pts_caveat': '224px cache PTS are metadata-derived, not newly measured; native/prior exact PTS are separately recorded. No f724 depth was invented.',
            'components_reference_correction': 'height/components.json selected_view_observability frame870 points to episode_18000. This audit independently uses episode_18001 frame870 from continuous manifest line413; it does not inherit that mismatched path.',
        },
        'preferred_next_action': next_action,
        'limits': {
            'new_video_decodes_this_finalization': 0,
            'existing_support_coverage_native_extraction': {'saved_images': 1, 'episode': 18001, 'frame': 724,
                                                          'cpu_codec_threads': 1, 'opens': 1, 'seeks': 1,
                                                          'frames_decoded_including_keyframe_preroll': 5,
                                                          'whole_episode_decoded': False},
            'gpu': False, 'downloads': False, 'runtime_modified': False, 'threshold_fitting': False,
            'object_motion_or_WAM_execution': False, 'new_model_research': False,
            'dev_report_GT_used_for_selection_or_labels': False,
            'human_visual_label': False, 'physical_success_label': None, 'physical_support_label': None,
            'new_physical_success': False, 'admission': False,
            'complete_envelope_certified': False,
        },
        'key_images': [str(OUT / n) for n in [
            'ep18001_f00724_head_native.png', 'ep18001_cache_contact_sheet_2.jpg',
            'ep18000_cache_contact_sheet_1.jpg', 'ep18000_cache_contact_sheet_2.jpg']],
        'validation': {'image_hashes_verified': len(verified_images), 'cache_byte_ranges_verified': len(cached['samples']),
                       'state_rows_verified': len(states), 'initial_head_depth_hashes_verified': len(exact),
                       'raw_video_payload_hashed_or_decoded_here': False},
    }
    assert len(verified_images) == 56
    path = OUT / 'coverage.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'result': str(path), 'bytes': path.stat().st_size,
                      'validation': result['validation'], 'new_video_decodes': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
