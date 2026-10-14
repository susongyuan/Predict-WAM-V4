"""Extra50 integrity, real DR/policy provenance, optimizer boundaries and physical diagnosis."""
from copy import deepcopy
from pathlib import Path
import ast
import json
import tempfile
import unittest
from unittest.mock import patch
from harness import zhuhai_dagger_training30k_v14 as d
from scripts import analyze_precision_dagger_milestones_v14 as analysis
from scripts import eval_precision_dagger_70k80k_fixed5_v14 as evaluator
from scripts.run_precision_dagger_recovery_branch_v14 import complete_five,after_trigger_stage


def manifest(rows):
    return dict(status='complete',success_count=100,requested_success_count=100,train_rows=rows,
        legacy_corrections=20,segmented_corrections=100,total_training_corrections=120,
        newly_added_segmented_episodes=50,quota_unit='accepted new segmented episodes',
        record_video=False,heldout_used=False)


def rule(step,grasp=False,placement=False):
    row=dict(control_step=step,objects={k:dict(touching_gripper=True) for k in analysis.ORDER},
        score=dict(full_task_success=False,items={k:dict(grasp_awarded=False,placement_valid=False,
        hold_seconds=0.,stable_seconds=0.) for k in analysis.ORDER}))
    row['score']['items']['bowl'].update(grasp_awarded=grasp,placement_valid=placement,
        hold_seconds=5. if grasp else 0.,stable_seconds=1. if placement else 0.)
    return row


class Regression(unittest.TestCase):
    def fixture(self,root):
        rows=[dict(seed=i,sha256=f'{i:064x}',path=str(root/f'episode_{i}.hdf5'),
            correction_variant='legacy_full_continuation' if i<20 else 'segmented_local_interventions') for i in range(120)]
        old=root/'retained.json';old.write_text(json.dumps(dict(train_rows=rows[:70])))
        for row in rows[70:]:
            raw=dict(scenario=dict(randomized=True),scene_manifest=dict(randomized=True),
                checkpoint_sha256='a'*64,policy_ready=dict(runtime=dict(checkpoint_step=50000,checkpoint_sha256='a'*64)),
                data_path=row['path'],data_sha256=row['sha256'])
            p=root/f'result_{row["seed"]}.json';p.write_text(json.dumps(raw))
            row.update(raw_result_path=str(p),raw_result_sha256=d.h.sha256_file(p),collection_policy_checkpoint_sha256='a'*64)
        m=manifest(rows);m['retained_collection']=dict(path=str(old),sha256=d.h.sha256_file(old));return m

    def test_exact_old70_plus_genuinely_new50_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=self.fixture(Path(tmp));self.assertEqual(d.validate_quota(m),dict(legacy=20,segmented=100,total=120))
            with patch.object(d.prior,'validate_rows',side_effect=lambda base,receipt:deepcopy(receipt['train_rows'])):
                self.assertEqual(len(d.validate_rows({},m)),120)
                bad=deepcopy(m);bad['train_rows'][0]['seed']=900
                with self.assertRaisesRegex(ValueError,'unchanged'):d.validate_rows({},bad)
                bad=deepcopy(m);bad['train_rows'][-1]['seed']=0
                with self.assertRaisesRegex(ValueError,'Duplicate'):d.validate_rows({},bad)
            bad=deepcopy(m);bad['train_rows'].pop()
            with self.assertRaisesRegex(ValueError,'genuinely new'):d.validate_quota(bad)

    def test_source_scene_or_wrong_policy_cannot_be_declared_DR50k(self):
        for change in ['source','wrong_policy']:
            with tempfile.TemporaryDirectory() as tmp:
                m=self.fixture(Path(tmp));row=m['train_rows'][70];p=Path(row['raw_result_path']);raw=json.loads(p.read_text())
                if change=='source':raw['scene_manifest']['randomized']=False
                else:raw['policy_ready']['runtime']['checkpoint_step']=45000
                p.write_text(json.dumps(raw));row['raw_result_sha256']=d.h.sha256_file(p)
                with patch.object(d.prior,'validate_rows',side_effect=lambda base,receipt:deepcopy(receipt['train_rows'])):
                    with self.assertRaisesRegex(ValueError,'actual50k'):d.validate_rows({},m)

    def test_30k_schedule_and_original_sampling_preserved(self):
        self.assertGreater(d.rate(50000),1e-6);self.assertAlmostEqual(d.rate(50099),3e-6)
        self.assertLess(d.rate(79999),1.00001e-6)
        for step in [49999,80000]:
            with self.assertRaises(ValueError):d.rate(step)
        self.assertIs(d.mixed_samples,d.prior.mixed_samples)

    def test_contract_preserves_original_statistics_conditions_splits_and_model(self):
        p=Path('outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger20k_segmented_20261011_v12/train/manifest.json')
        base=json.loads(p.read_text())['contract'];before=deepcopy(base)
        rows=[dict(correction_variant='legacy_full_continuation')]*20+[dict(correction_variant='segmented_local_interventions')]*100
        out=d.contract(base,rows,parent=dict(step=50000,model_only=False),collection={},sources={})
        self.assertEqual(base,before)
        for key in ['statistics','idm_statistics','conditions','splits','precision','numerical_normalization','precision_cache_compatibility']:
            self.assertEqual(out[key],base[key])
        self.assertEqual(out['config']['model'],base['config']['model'])
        self.assertEqual((out['dagger_training']['start_step'],out['dagger_training']['target_step'],out['dagger_training']['total_train_episodes']),(50000,80000,280))

    def test_actual_loss_backward_and_optimizer_update_sequence_unchanged(self):
        def steps(path):
            tree=ast.parse(Path(path).read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
            loop=next(n for n in ast.walk(execute) if isinstance(n,ast.While));nodes=[]
            for n in loop.body:
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='row' for t in n.targets):break
                nodes.append(n)
            return ast.dump(ast.Module(body=nodes,type_ignores=[]))
        self.assertEqual(steps('scripts/train_precision_dagger20k_segmented_v7.py'),steps('scripts/train_precision_dagger30k_v14.py'))

    def test_gripper_contact_does_not_mean_grasp_and_lost_placement_is_detected(self):
        result=analysis.progress([rule(0),rule(1)])
        self.assertEqual(result['failure_stage'],'before_first_verified_grasp')
        self.assertEqual(result['objects']['bowl']['gripper_contact_frames'],2)
        self.assertEqual(analysis.progress([rule(0),rule(1,True)])['failure_stage'],'bowl_placement')
        self.assertEqual(analysis.progress([rule(0),rule(1,True,True),rule(2,True,False)])['failure_stage'],'bowl_lost_valid_placement')
        with self.assertRaises(ValueError):analysis.progress([rule(1),rule(0)])

    def test_five_test_results_and_no_evaluation_during_live_training(self):
        rows=[dict(seed=s,full_task_success=i<1) for i,s in enumerate(evaluator.SEEDS)]
        cohort=dict(status='completed',checkpoint=dict(step=70000),seeds=list(evaluator.SEEDS),planned_episodes=5,
            episodes=rows,full_successes=1,success_rate=.2)
        self.assertTrue(complete_five(cohort,70000))
        self.assertFalse(complete_five(dict(cohort,full_successes=2),70000))
        cm=manifest([dict(correction_variant='legacy_full_continuation')]*20+[dict(correction_variant='segmented_local_interventions')]*100)
        tm=dict(status='completed',step=80000,additional_updates=30000,new_segmented_episodes=100,
            legacy_episodes=20,final_checkpoint_integrity_verified=True)
        self.assertEqual(after_trigger_stage(cm,tm,[dict(MainPID='123')]),'waiting_for_training_exit')
        self.assertEqual(after_trigger_stage(cm,tm,[dict(MainPID='0')]),'evaluations_ready')
        with self.assertRaises(ValueError):after_trigger_stage(dict(cm,status='halted'),tm,[])

    def test_final_two_milestones_are_exactly70k80k_and_denominator5(self):
        plan=dict(checkpoint=dict(step=70000,sha256='a'*64),action_postprocessing_sha256='b'*64,baseline={},interpretation='paired five')
        rows=[dict(seed=s,full_task_success=i<1) for i,s in enumerate(evaluator.SEEDS)]
        s=evaluator.summary(plan,rows,'completed');self.assertEqual((s['checkpoint_step'],s['planned_episodes'],s['success_rate']),(70000,5,.2))
        self.assertIsNone(evaluator.summary(plan,rows[:1],'evaluating')['success_rate'])
        with self.assertRaisesRegex(ValueError,'70k/80k'):
            evaluator.metadata_milestone('unused',75000)


if __name__=='__main__':unittest.main()
