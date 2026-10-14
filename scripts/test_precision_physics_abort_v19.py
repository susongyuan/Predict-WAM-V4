"""Regression checks for collision failure accounting, without physics retries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts import precision_physics_abort_evidence_v19 as c
from scripts import eval_precision_dagger_milestone_fixed10_v19 as fixed
from scripts import eval_precision_dagger_70k80k_fixed5_v19 as future
from scripts import analyze_precision_dagger_milestones_v19 as analysis

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger_45k_fixed10_20261011_v18/zhuhai_task2/seed_932101'


class CollisionEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules=[json.loads(l) for l in (CASE/'rule_evidence.jsonl').read_text().splitlines()]
        cls.previous=cls.rules[-2];cls.terminal=cls.rules[-1]

    def test_actual_contact_interval(self):
        self.assertEqual(c.terminal_contact(self.terminal,self.previous)['base_score'],0)
        self.assertEqual(self.previous['control_step'],651)
        self.assertEqual(self.terminal['control_step'],652)

    def test_missing_or_unknown_contact_is_rejected(self):
        for events in ([],[dict(self.terminal['physics_safety']['events'][0],reason='unknown')]):
            rule=copy.deepcopy(self.terminal);rule['physics_safety']['events']=events
            with self.assertRaises(c.e.EvidenceError):c.terminal_contact(rule,self.previous)

    def test_earlier_unsafe_frame_is_rejected(self):
        previous=copy.deepcopy(self.previous);previous['physics_safety']['passed']=False
        with self.assertRaises(c.e.EvidenceError):c.terminal_contact(self.terminal,previous)

    def test_stale_event_clock_and_monitor_change_are_rejected(self):
        rule=copy.deepcopy(self.terminal);rule['physics_safety']['events'][0]['time_seconds']=1.
        with self.assertRaises(c.e.EvidenceError):c.terminal_contact(rule,self.previous)
        rule=copy.deepcopy(self.terminal);rule['physics_safety']['checks']=[]
        with self.assertRaises(c.e.EvidenceError):c.terminal_contact(rule,self.previous)

    def test_unknown_infrastructure_still_halts(self):
        raw=json.loads((CASE/'result.json').read_text());raw['error']='InfrastructureError: lost worker'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json';path.write_text(json.dumps(raw))
            with self.assertRaises(c.e.EvidenceError):fixed.classify(path,{}, {},'unused')

    def test_failure_summary_and_future_seed_denominators(self):
        plan=dict(checkpoint={'step':45000,'sha256':'a'*64},action_postprocessing_sha256='b'*64,
            baseline={},interpretation='test',adopted_attempts=[{'row':{'seed':932101}}],
            physical_failure_accounting='one physical failure')
        row=dict(seed=932101,status='physics_safety_abort',full_task_success=False)
        summary=fixed.summary(plan,[row],'evaluating')
        self.assertIsNone(summary['success_rate']);self.assertEqual(summary['full_successes'],0)
        self.assertEqual(summary['planned_episodes'],10)
        self.assertEqual(summary['adopted_original_seeds'],[932101])
        self.assertEqual(future.SEEDS,tuple(range(932101,932106)))

    def test_offline_analysis_excludes_unsafe_terminal(self):
        safe=analysis.progress(self.rules[:-1])
        self.assertEqual(safe['last_control_step'],651)
        self.assertTrue(all(not v['ever_verified_grasp'] for v in safe['objects'].values()))


if __name__=='__main__':unittest.main()
