"""Final evaluation cannot race incomplete quotas, updates, or live training."""
from copy import deepcopy
import unittest
from scripts.supervise_precision_dagger50k_v9 import next_stage
from scripts.eval_precision_dagger50k_fixed5_v9 import training_finished, metadata50
from scripts.test_precision_dagger_segmented_quota import manifest


def complete_collection():
    rows=[dict(correction_variant='legacy_full_continuation') for _ in range(20)]
    rows += [dict(correction_variant='segmented_local_interventions') for _ in range(50)]
    return manifest(rows)


def completed_training():
    return dict(status='completed',step=50000,additional_updates=20000,new_segmented_episodes=50,
                legacy_episodes=20,final_checkpoint_integrity_verified=True)


class Tests(unittest.TestCase):
    def test_test_is_blocked_until_full_quota_and_all_new_updates(self):
        units=[dict(MainPID='0',ActiveState='inactive')]*2
        collection=complete_collection();training=completed_training()
        self.assertEqual(next_stage(dict(collection,status='parallel_collecting'),training,units),'collecting')
        self.assertEqual(next_stage(collection,dict(training,status='training_child_running',step=49000),units),'training')
        self.assertEqual(next_stage(collection,training,units),'final_evaluation_ready')
        bad=deepcopy(collection);bad['train_rows'].pop()
        with self.assertRaisesRegex(ValueError,'50 new segmented'):next_stage(bad,training,units)
        for field,value in [('step',49000),('additional_updates',19999),('final_checkpoint_integrity_verified',False)]:
            self.assertFalse(training_finished(dict(training,**{field:value})))

    def test_successful_status_cannot_overlap_a_live_trainer(self):
        units=[dict(MainPID='0',ActiveState='inactive'),dict(MainPID='123',ActiveState='active')]
        self.assertEqual(next_stage(complete_collection(),completed_training(),units),'waiting_for_training_process_exit')

    def test_halted_pipeline_and_old_milestone_are_never_tested_as_final50k(self):
        with self.assertRaisesRegex(ValueError,'stage halted'):
            next_stage(dict(status='halted',error='retained expert guard'),dict(status='waiting_for_collection'),[])
        with self.assertRaisesRegex(ValueError,'50000 model-only'):
            metadata50('outputs/zhuhai2026_tasks12/wam_precision_v2_task2_30k_20261010_v1/train/checkpoint_step_030000.pt')


if __name__=='__main__':unittest.main()
