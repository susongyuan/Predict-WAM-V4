"""CPU fixtures: task/count binding and immutable TRAIN retrieval, not physical admission."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from harness.robotwin_policy_worker import sha256, InfrastructureError
from harness.robotwin_vtt_runtime import VTTMonitor
from harness.robotwin_condition_memory import DemoConditionMemory


class TaskVTTTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.bank=Path(self.temp.name)/'bank';self.bank.mkdir()
        self.adapter=Path(self.temp.name)/'adapter';self.adapter.mkdir()
        np.save(self.bank/'prototypes.npy',np.ones((1,1024),np.float32))
        np.save(self.bank/'action_vtt_prototype.npy',np.ones(1024,np.float32))
        np.save(self.bank/'vtt_instances.npy',np.ones((40,1024),np.float32))
        np.save(self.bank/'context_cls.npy',np.ones((40,1024),np.float32))
        result=dict(task='put_bottles_dustbin',TRAIN_episodes=40,records=[dict(episode=f'train{i}') for i in range(40)],
                    done_progress=.9,done_alignment=.8,min_steps=30,completed=True,heldout_used=False,
                    source_split='train',split_sha256='fake-split',cache_id='fake-cache')
        (self.bank/'result.json').write_text(json.dumps(result))
        protocol=dict(task_name='put_bottles_dustbin',split_sha256='fake-split',feature_cache_id='fake-cache')
        for name,key in [('action_vtt_prototype.npy','VTT_prototype_sha256'),('vtt_instances.npy','VTT_instances_sha256'),
                         ('context_cls.npy','VTT_context_sha256'),('result.json','VTT_bank_result_sha256')]:
            protocol[key]=sha256(self.bank/name)
        (self.adapter/'protocol.json').write_text(json.dumps(protocol))

    def test_forty_train_supported_and_closure_remains_advisory(self):
        monitor=VTTMonitor(self.bank,task='put_bottles_dustbin',train_episodes=40)
        monitor.observe(np.zeros(1024,np.float32),0)
        result=monitor.observe(np.ones(1024,np.float32),40)
        self.assertTrue(result['closure_candidate']);self.assertIsNone(result['task_success_verified'])
        self.assertFalse(monitor.identity['physical_verifier_admitted'])

    def test_wrong_task_and_count_rejected(self):
        for args in [dict(task='put_object_cabinet'),dict(train_episodes=400)]:
            with self.assertRaises(InfrastructureError):VTTMonitor(self.bank,**args)
            with self.assertRaises(InfrastructureError):DemoConditionMemory(self.bank,self.adapter/'model.pt',**args)

    def test_retrieval_no_replacement_and_no_weight_updates(self):
        memory=DemoConditionMemory(self.bank,self.adapter/'model.pt',task='put_bottles_dustbin',train_episodes=40)
        names=[memory.select(np.ones(1024,np.float32))[1] for _ in range(40)]
        self.assertEqual(len(set(names)),40)
        with self.assertRaises(InfrastructureError):memory.select(np.ones(1024,np.float32))

    def test_changed_context_metadata_rejected(self):
        np.save(self.bank/'context_cls.npy',np.zeros((40,1024),np.float32))
        with self.assertRaises(InfrastructureError):DemoConditionMemory(self.bank,self.adapter/'model.pt',task='put_bottles_dustbin')

    def test_legacy_default_cabinet_count_still_supported(self):
        result=json.loads((self.bank/'result.json').read_text());result.pop('task');result['TRAIN_episodes']=400
        (self.bank/'result.json').write_text(json.dumps(result))
        monitor=VTTMonitor(self.bank,task='put_object_cabinet',train_episodes=400)
        self.assertEqual(monitor.identity['TRAIN_episodes'],400)


if __name__=='__main__':unittest.main()
