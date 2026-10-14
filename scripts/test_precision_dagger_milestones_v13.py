"""Paired complete ten-scene cohorts, exact milestone identity and conditional branch."""
from copy import deepcopy
from pathlib import Path
import ast
import unittest
from scripts import eval_precision_dagger_milestone_fixed10_v13 as evaluator
from scripts.supervise_precision_dagger_milestones_v13 import complete_cohort,conditional_recovery_required


def cohort(step,successes=0):
    return dict(status='completed',checkpoint=dict(step=step),planned_episodes=10,
        seeds=list(evaluator.SEEDS),episodes=[dict(seed=s,full_task_success=i<successes)
        for i,s in enumerate(evaluator.SEEDS)],full_successes=successes,success_rate=successes/10)


class Regression(unittest.TestCase):
    def test_exact_ten_paired_scenes_and_latest_checkpoint_failure_trigger(self):
        cohorts={'45000':cohort(45000,1),'50000':cohort(50000)}
        self.assertTrue(conditional_recovery_required(cohorts))
        cohorts['50000']=cohort(50000,1)
        self.assertFalse(conditional_recovery_required(cohorts))
        cohorts['50000']['episodes'].pop()
        with self.assertRaises(ValueError):conditional_recovery_required(cohorts)

    def test_incomplete_reordered_duplicate_or_wrong_checkpoint_never_complete(self):
        row=cohort(45000);self.assertTrue(complete_cohort(row,45000))
        for changed in [dict(status='evaluating'),dict(planned_episodes=5),dict(checkpoint=dict(step=50000)),
            dict(full_successes=1),dict(success_rate=.2)]:
            self.assertFalse(complete_cohort(dict(row,**changed),45000))
        row['episodes'][-1]['seed']=row['episodes'][0]['seed']
        self.assertFalse(complete_cohort(row,45000))

    def test_summary_denominator_ten_and_milestone_45_preserved(self):
        plan=dict(checkpoint=dict(step=45000,sha256='a'*64),action_postprocessing_sha256='b'*64,
            baseline={},interpretation='paired source development scenes')
        rows=cohort(45000,2)['episodes'];out=evaluator.summary(plan,rows,'completed')
        self.assertEqual((out['checkpoint_step'],out['planned_episodes'],out['success_rate']),(45000,10,.2))
        self.assertIsNone(evaluator.summary(plan,rows[:2],'evaluating')['success_rate'])
        with self.assertRaises(ValueError):evaluator.summary(plan,rows[:5],'completed')

    def test_classifier_unchanged_except_admitted_seed_list(self):
        def function(path,normalize=False):
            source=Path(path).read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='classify')
            text=ast.get_source_segment(source,node)
            if normalize:text=text.replace('evidence.SEEDS','SEEDS').replace('evidence.','e.')
            return ast.dump(ast.parse(text))
        self.assertEqual(function('scripts/wait_precision_official_braking.py',True),
            function('scripts/eval_precision_dagger_milestone_fixed10_v13.py'))

    def test_wrong_model_milestone_rejected(self):
        path='outputs/zhuhai2026_tasks12/wam_precision_v2_task2_30k_20261010_v1/train/checkpoint_step_030000.pt'
        with self.assertRaisesRegex(ValueError,'requested model-only'):evaluator.metadata_milestone(path,45000)
        with self.assertRaisesRegex(ValueError,'45k/50k'):evaluator.metadata_milestone(path,35000)


if __name__=='__main__':unittest.main()
