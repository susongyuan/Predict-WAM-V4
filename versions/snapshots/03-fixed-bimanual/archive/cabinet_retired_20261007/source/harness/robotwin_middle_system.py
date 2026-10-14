"""No-VLM/ASK middle/lower execution with measured controller supervision.

Includes transactional dispatch, robot feedback, bounded HOLD/clear/reobserve,
TRAIN condition retrieval and scoped controller-outcome memory. The VTT closure
nomination never certifies grasp/containment and task outcomes remain unknown.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from harness.core import Memory, Scope, context_fingerprint
from harness.robotwin_policy_worker import envelope
from harness.robotwin_robot_feedback import robot_feedback, hold_measured
from harness.robotwin_system_adapter import RobotwinSystemAdapter, reject_gt


class MotionWindow:
    def __init__(self, size=41):
        self.rows = deque(maxlen=size)

    def push(self, step, feedback, progress=None):
        reject_gt(feedback)
        actual = np.array([q for side in ('left', 'right')
                           for q in feedback['arms'][side]['actual_qpos']], dtype=float)
        commands = np.array([q for side in ('left', 'right')
                            for q in feedback['arms'][side]['drive_targets']], dtype=float)
        if (actual.shape != (12,) or commands.shape != (12,)
                or not np.isfinite(actual).all() or not np.isfinite(commands).all()
                or progress is not None and not np.isfinite(progress)):
            raise ValueError('finite native12 measurements required')
        if self.rows and step != self.rows[-1][0] + 1:
            raise ValueError('contiguous measured motion window required')
        self.rows.append((step, actual, commands, progress))

    def metrics(self):
        if len(self.rows) != self.rows.maxlen:
            return None
        actual = np.stack([r[1] for r in self.rows])
        commands = np.stack([r[2] for r in self.rows])
        progress = [r[3] for r in self.rows]
        a = float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum())
        c = float(np.linalg.norm(np.diff(commands, axis=0), axis=1).sum())
        p = float(np.ptp(progress)) if all(v is not None for v in progress) else None
        return {'first_step': self.rows[0][0], 'last_step': self.rows[-1][0],
                'actual_joint_path_rad': a, 'command_joint_path_rad': c,
                'VTT_progress_range': p,
                'controller_stalled': bool(a < .05 and c < .1 and (p is None or p < .025)),
                'motion_resumed': bool(a >= .05)}


class MiddleSystem(RobotwinSystemAdapter):
    def __init__(self, client, driver, *, robot, refresh, out, episode, with_vtt):
        super().__init__(client, driver, max_age_s=30)
        self.robot, self.refresh, self.out, self.with_vtt = robot, refresh, Path(out), with_vtt
        self.window = MotionWindow()
        self.progress = None
        self.retry_count = 0
        self.pending_recovery = None
        self.memory = Memory(self.out / 'controller_experience.sqlite', split_id='dev_middle', run_id=episode)
        scope_cfg = {'episode': episode, 'with_vtt': with_vtt,
                     'window': 40, 'actual_path_rad': .05, 'command_path_rad': .1,
                     'earliest_retry': 160, 'max_retries': 2}
        self.scope = Scope('put_object_cabinet', 'controller_motion', context_fingerprint(scope_cfg),
                           'aloha_native14', 'native_TOPP', 'cabinet30k_vtt_residual_v1' if with_vtt else 'cabinet30k_zero',
                           'measured_joint_motion_v1', 'dev_middle', episode)
        self.report = {'configuration': 'middle_measured_feedback_v1', 'VLM': False, 'ASK': False,
                       'with_vtt': with_vtt, 'supervisor_controls_actions': True,
                       'physical_task_verifier_admitted': False, 'task_success_verified': None,
                       'certified_full_middle_system': False, 'recovery_attempts': 0,
                       'TRAIN_memory_condition_selections': 0, 'controller_memory_records': 0,
                       'task_success_memory_records': 0, 'VTT_closure_candidates': 0,
                       'intervention_basis': 'measured_controller_stall_not_empty_grasp_or_slip',
                       'threshold_status': 'development_controller_threshold_not_grasp_verifier'}

    def event(self, value):
        with (self.out / 'middle_events.jsonl').open('a') as f:
            f.write(json.dumps(value, allow_nan=False) + '\n')

    def record_motion(self, step, outcome, recovery_of=None):
        record = self.memory.record(self.scope, predicates={'controller_motion_resumed': outcome},
                     terminal=True, episode_id=self.episode,
                     source=str(self.out / 'middle_events.jsonl') + '#step' + str(step),
                     verifier_kind='controller_postconditions', observation_id=f'{self.episode}:motion:{step}',
                     recovery_of=recovery_of)
        self.report['controller_memory_records'] += 1
        return record

    def prepare(self, episode, step, observation_id, observed_at, observation):
        # No inspection or hold may precede the transaction lifecycle check.
        if self.failed or self.pending is not None or episode != self.episode or step != self.next_step:
            raise ValueError('inactive/pending/wrong transaction before supervision')
        reject_gt(observation)
        feedback = robot_feedback(self.robot())
        self.window.push(step, feedback, self.progress)
        metrics = self.window.metrics()
        self.event({'step': step, 'kind': 'robot_feedback', 'observation_id': observation_id,
                    'feedback': feedback, 'motion_window': metrics, 'task_outcome': None})
        if self.pending_recovery and metrics and metrics['first_step'] >= self.pending_recovery['step']:
            outcome = metrics['motion_resumed']
            record = self.record_motion(step, outcome, self.pending_recovery['record'] if outcome else None)
            self.event({'step': step, 'kind': 'controller_retry_postcondition', 'record': record,
                        'motion_resumed': outcome, 'metrics': metrics, 'task_success': None})
            self.pending_recovery = None
        recalls = self.memory.recall(self.scope, policy_only=True)
        verified_controller_failures = sum(r['outcome'] == 'failure' for r in recalls)
        if (step >= 160 and metrics and metrics['controller_stalled'] and self.pending_recovery is None
                and self.retry_count < 2 and verified_controller_failures < 4):
            failure_record = self.record_motion(step, False)
            hold = hold_measured(self.robot())
            clear = self.replan('measured_controller_stall_fresh_reobserve')
            selected = None
            if self.with_vtt:
                selected = self.client.request(envelope('next_demo_condition', episode, step), 'condition_selected')
                if selected.get('history_preserved') is not True or selected.get('next_step') != step:
                    raise RuntimeError('condition selection failed history/step binding')
                self.report['TRAIN_memory_condition_selections'] += 1
            observation, observed_at = self.refresh()
            self.retry_count += 1
            self.report['recovery_attempts'] = self.retry_count
            self.pending_recovery = {'step': step, 'record': failure_record}
            self.event({'step': step, 'kind': 'supervisor_hold_clear_reobserve',
                        'controller_failure_record': failure_record,
                        'memory_recalled_records': len(recalls), 'hold': hold, 'clear': clear,
                        'condition_selection': selected, 'history_reset': False,
                        'grasp_failure_verified': None})
        reply = super().prepare(episode, step, observation_id, observed_at, observation)
        vtt = reply.get('vtt')
        if vtt:
            if vtt.get('episode') != episode or vtt.get('observation_step') != step:
                raise ValueError('stale VTT observation')
            self.progress = vtt['progress']
            if vtt['closure_candidate']:
                self.report['VTT_closure_candidates'] += 1
            self.event({'step': step, 'kind': 'VTT_nomination', 'vtt': vtt,
                        'supervisor_completion_decision': 'UNKNOWN_continue', 'task_memory_written': False})
        return reply

    def close(self):
        self.memory.close()
