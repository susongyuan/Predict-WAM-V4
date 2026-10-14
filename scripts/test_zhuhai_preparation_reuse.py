#!/usr/bin/env python3
"""Small synthetic CPU tests. Never invoke real statistics/VTT fitting or models."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from harness import zhuhai_preparation_reuse as reuse
from harness import zhuhai_training as zt


class NoFeatureReads:
    condition = None
    def __getattr__(self, name):
        raise AssertionError(f'VAL/heldout features or data must not be read: {name}')


def config():
    return dict(**zt.protocol_binding(),
        common=dict(action_dim=16, state_dim=16, action_chunk_size=32, action_execution_horizon=8,
                    num_inference_steps=10, proprio_len=7),
        dataset=dict(frequency_hz=10., future_frame_offset=32, image_size=[224, 224]),
        model=dict(proprio_mask='native_all16', views=list(zt.MODEL_VIEWS),
                   v3=dict(proprio_lags=[2,4,8,16,32], memory_lags=[4,16]),
                   vision_encoder=dict(feat_layers=[-12,-8,-4])),
        training=dict(max_steps=2, batch_size=1, optimizer='tiny-test'), mode='tiny', out='ignored_tiny')


def source(directory, cfg, task, seed):
    path = directory / f'{task}_{seed}.hdf5'
    state = np.arange(6*16,dtype=np.float32).reshape(6,16)/1000 + seed/10000
    state[:,[7,15]] = .6
    action = state.copy(); action[:,:7] += .01; action[:,[7,15]] = .7
    with h5py.File(path,'x') as handle:
        handle.attrs['metadata'] = json.dumps(dict(synthetic_cpu_fixture=True,contract=zt.protocol_binding()))
        offset = 0
        for field, width in zip(reuse._FIELDS,(7,1,7,1)):
            handle.create_dataset('state/'+field,data=state[:,offset:offset+width])
            handle.create_dataset('action/'+field,data=action[:,offset:offset+width])
            offset += width
    sha = zt.sha256_file(path)
    row = dict(split='train',path=str(path),sha256=sha,seed=seed,family_id=f'{task}-{seed}',**zt.protocol_binding())
    ep = SimpleNamespace(path=path,row=row,sha256=sha,state=state,action=action)
    return SimpleNamespace(episodes=[ep],proprio_lags=list(cfg['model']['v3']['proprio_lags']),
                           memory_lags=list(cfg['model']['v3']['memory_lags']),condition=None)


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='zhuhai-preparation-reuse-cpu-')
        self.directory = Path(self.tmp.name)
        self.root = self.directory/'reuse'
        self.cfg = config()
        self.train = {task:source(self.directory,self.cfg,task,101+i) for i,task in enumerate(('zhuhai_task1','zhuhai_task2'))}
        self.all = dict(train=self.train,
            val={task:NoFeatureReads() for task in self.train},heldout={task:NoFeatureReads() for task in self.train})
        self.feature = dict(encoder_sha256='a'*64,preprocess_sha256='b'*64)
        self.counts = dict(statistics=0,conditions=0)
        self.cache = reuse.PreparationReuse(self.root,lock_timeout=.1)
        self.protected = {name:zt.sha256_file(zt.ROOT/name) for name in reuse._SOURCES}

    def tearDown(self):
        for name, expected in self.protected.items():
            self.assertEqual(zt.sha256_file(zt.ROOT/name),expected)
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('isaacsim',sys.modules)
        self.assertLess(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,500*1024**2)
        self.tmp.cleanup()

    def stats_builder(self, datasets, cfg, *, feature_identity, check=lambda: None):
        # Synthetic moments and constant synthetic pool std, not the real builder.
        self.counts['statistics'] += 1
        check()
        provenance = zt.train_provenance(datasets,cfg)
        provenance.update(feature_identity=copy.deepcopy(feature_identity),variance='population_float64_parallel_welford',
                          std_floor=1e-3,history_start='episode_frame_0')
        def moments(kind):
            values=np.concatenate([getattr(ep,kind) for ds in datasets.values() for ep in ds.episodes]).astype(np.float64)
            return {k:v.tolist() for k,v in dict(min=values.min(0),max=values.max(0),mean=values.mean(0),std=np.maximum(values.std(0),1e-3)).items()}
        v3=cfg['model']['v3']
        return dict(normalization=dict(schema_version=2,action_repr='absolute',provenance=provenance,
                    **{zt.NORMALIZATION_KEY:dict(state=moments('state'),action=moments('action'))}),
            temporal_stats=dict(schema_version=2,views=list(cfg['model']['views']),proprio_lags=list(v3['proprio_lags']),
                memory_lags=list(v3['memory_lags']),proprio_diff_std=np.ones((len(v3['proprio_lags'])+1,16)).tolist(),
                memory_diff_std=np.ones((3,len(v3['memory_lags'])+1,2048)).tolist(),provenance=provenance),
            proprio_keep_mask=[1.]*16,**zt.protocol_binding())

    def bank_builder(self, datasets, cfg, statistics):
        # Deliberately no check keyword: keep original install_conditions signature.
        self.counts['conditions'] += 1
        vectors={task:np.sin(np.arange(1024,dtype=np.float32)/(17+i)).tolist() for i,task in enumerate(sorted(datasets['train']))}
        return dict(format=zt.FORMAT+'/vtt',source_split='train',conditions=vectors,
            provenance=zt.train_provenance(datasets['train'],cfg),statistics_sha256=zt.digest(statistics),
            feature_identity=statistics['normalization']['provenance']['feature_identity'],**zt.protocol_binding())

    def statistics(self, cache=None, cfg=None, feature=None, builder=None, train=None):
        return (cache or self.cache).statistics(train or self.train,cfg or self.cfg,
            feature_identity=feature or self.feature,builder=builder or self.stats_builder)

    def conditions(self, statistics, cache=None, builder=None):
        return (cache or self.cache).conditions(self.all,self.cfg,statistics,builder=builder or self.bank_builder)

    def test_tiny_to_train_reuses_without_run_config_keys(self):
        first=self.statistics();key=self.cache.receipt()['operations'][-1]['key']
        first_bank=self.conditions(first)
        cfg=copy.deepcopy(self.cfg)
        cfg.update(mode='train',out='different/full/run',optimizer={'lr':.9})
        cfg['training'].update(max_steps=5000,batch_size=4,mode='train',out='other',optimizer='different')
        cold=reuse.PreparationReuse(self.root)
        second=self.statistics(cache=cold,cfg=cfg)
        self.assertEqual(first,second)
        self.assertEqual(self.counts['statistics'],1)
        self.assertEqual(cold.receipt()['operations'][-1]['key'],key)
        self.assertEqual(cold.receipt()['operations'][-1]['outcome'],'hit')
        bank=cold.conditions(self.all,cfg,second,builder=self.bank_builder)
        self.assertEqual(bank,first_bank)
        self.assertEqual(self.counts,{'statistics':1,'conditions':1})
        self.assertEqual(cold.receipt()['operations'][-1]['outcome'],'hit')
        json.dumps(cold.receipt(),allow_nan=False)

    def test_encoder_preprocess_history_and_algorithm_separate_keys(self):
        self.statistics();keys={self.cache.receipt()['operations'][-1]['key']}
        for field in ('encoder_sha256','preprocess_sha256'):
            fi=dict(self.feature);fi[field]='c'*64
            self.statistics(feature=fi);keys.add(self.cache.receipt()['operations'][-1]['key'])
        cfg=copy.deepcopy(self.cfg);cfg['model']['v3']['memory_lags']=[2,8]
        train=copy.deepcopy(self.train)
        for ds in train.values():ds.memory_lags=[2,8]
        self.statistics(cfg=cfg,train=train);keys.add(self.cache.receipt()['operations'][-1]['key'])
        hashes=reuse._source_hashes();hashes['harness/zhuhai_training.py']='d'*64
        with mock.patch.object(reuse,'_source_hashes',return_value=hashes):
            self.statistics();keys.add(self.cache.receipt()['operations'][-1]['key'])
        self.assertEqual(len(keys),5)
        self.assertEqual(self.counts['statistics'],5)

    def test_source_content_change_separates_but_stale_or_polluted_input_rejects(self):
        self.statistics();original=self.cache.receipt()['operations'][-1]['key']
        changed=dict(self.train);changed['zhuhai_task1']=source(self.directory,self.cfg,'zhuhai_task1',201)
        self.statistics(train=changed)
        self.assertNotEqual(self.cache.receipt()['operations'][-1]['key'],original)
        episode=self.train['zhuhai_task1'].episodes[0]
        original_state=episode.state.copy()
        episode.state[0,0]+=1
        with self.assertRaises(ValueError):self.statistics()
        episode.state[:]=original_state
        with h5py.File(episode.path,'r+') as handle:handle.attrs['tampered']=True
        with self.assertRaises(ValueError):self.statistics()

    def test_bank_hit_reassigns_all_splits_without_feature_reads(self):
        stats=self.statistics();first=self.conditions(stats)
        for split in self.all.values():
            for ds in split.values():ds.condition=None
        cold=reuse.PreparationReuse(self.root)
        stats=self.statistics(cache=cold)
        with mock.patch.object(zt,'validate_statistics',wraps=zt.validate_statistics) as vs, \
             mock.patch.object(zt,'validate_conditions',wraps=zt.validate_conditions) as vc:
            second=self.conditions(stats,cache=cold)
            self.assertGreater(vs.call_count,0);self.assertGreater(vc.call_count,0)
        self.assertEqual(first,second);self.assertEqual(self.counts,{'statistics':1,'conditions':1})
        for split in self.all.values():
            for task,ds in split.items():np.testing.assert_array_equal(ds.condition,np.asarray(first['conditions'][task],np.float32))
        self.all['val']['zhuhai_task1'].condition[0]+=10
        self.assertNotEqual(self.all['val']['zhuhai_task1'].condition[0],self.train['zhuhai_task1'].condition[0])

    def test_payload_hash_tamper_rejected_without_rebuild_or_deletion(self):
        self.statistics();attempt=Path(self.cache.receipt()['operations'][-1]['attempt']);payload=attempt/'payload.json'
        value=json.loads(payload.read_text());value['normalization'][zt.NORMALIZATION_KEY]['state']['mean'][0]+=100
        payload.write_bytes(reuse._bytes(value));bad=payload.read_bytes()
        with self.assertRaises(ValueError):self.statistics(cache=reuse.PreparationReuse(self.root))
        self.assertEqual(self.counts['statistics'],1);self.assertEqual(payload.read_bytes(),bad)

    def test_matching_hashes_do_not_bypass_train_contract_validation(self):
        self.statistics();attempt=Path(self.cache.receipt()['operations'][-1]['attempt'])
        payload=attempt/'payload.json';receipt=attempt/'receipt.json';pointer=attempt.parent/'COMMITTED.json'
        value=json.loads(payload.read_text())
        for section in ('normalization','temporal_stats'):value[section]['provenance']['fit_split']='heldout'
        payload.write_bytes(reuse._bytes(value))
        rec=json.loads(receipt.read_text());rec['payload_sha256']=zt.sha256_file(payload);receipt.write_bytes(reuse._bytes(rec))
        marker=json.loads(pointer.read_text());marker['receipt_sha256']=zt.sha256_file(receipt);pointer.write_bytes(reuse._bytes(marker))
        with self.assertRaises(ValueError):self.statistics()
        self.assertEqual(self.counts['statistics'],1)

    def test_tampered_bank_is_validated_before_any_split_assignment(self):
        stats=self.statistics();self.conditions(stats)
        attempt=Path(self.cache.receipt()['operations'][-1]['attempt'])
        payload,receipt,pointer=attempt/'payload.json',attempt/'receipt.json',attempt.parent/'COMMITTED.json'
        value=json.loads(payload.read_text());value['conditions']['zhuhai_task1']=[1.]*1024
        payload.write_bytes(reuse._bytes(value))
        record=json.loads(receipt.read_text());record['payload_sha256']=zt.sha256_file(payload);receipt.write_bytes(reuse._bytes(record))
        marker=json.loads(pointer.read_text());marker['receipt_sha256']=zt.sha256_file(receipt);pointer.write_bytes(reuse._bytes(marker))
        for split in self.all.values():
            for ds in split.values():ds.condition=None
        with self.assertRaises(ValueError):self.conditions(stats,cache=reuse.PreparationReuse(self.root))
        self.assertTrue(all(ds.condition is None for split in self.all.values() for ds in split.values()))
        self.assertEqual(self.counts['conditions'],1)

    def test_builder_mutating_train_content_cannot_publish(self):
        def poisoned(datasets,cfg,*,feature_identity,check=lambda:None):
            datasets['zhuhai_task1'].episodes[0].action[0,0]+=1
            return self.stats_builder(datasets,cfg,feature_identity=feature_identity,check=check)
        with self.assertRaises(ValueError):self.statistics(builder=poisoned)
        self.assertFalse(list(self.root.glob('*/statistics/COMMITTED.json')))
        self.assertEqual(len(list(self.root.glob('*/statistics/attempt-*/failed.json'))),1)

    def test_conditions_reject_uncommitted_or_mutated_statistics(self):
        direct=self.stats_builder(self.train,self.cfg,feature_identity=self.feature)
        with self.assertRaises(ValueError):self.conditions(direct)
        stats=self.statistics();stats['normalization'][zt.NORMALIZATION_KEY]['state']['mean'][0]+=1
        with self.assertRaises(ValueError):self.conditions(stats)
        self.assertEqual(self.counts['conditions'],0)

    def test_val_contamination_rejected(self):
        changed=copy.deepcopy(self.train);changed['zhuhai_task1'].episodes[0].row['split']='val'
        with self.assertRaises(ValueError):self.statistics(train=changed)
        self.assertEqual(self.counts['statistics'],0)

    def test_exception_releases_lock_and_preserves_attempt(self):
        def interrupted(*args,**kwargs):raise KeyboardInterrupt('CPU synthetic interruption')
        with self.assertRaises(KeyboardInterrupt):self.statistics(builder=interrupted)
        attempt=Path(self.cache.receipt()['operations'][-1]['attempt'])
        before=(attempt/'failed.json').read_bytes()
        result=self.statistics(cache=reuse.PreparationReuse(self.root,lock_timeout=0))
        zt.validate_statistics(result,self.cfg)
        self.assertEqual((attempt/'failed.json').read_bytes(),before)
        self.assertTrue((attempt/'started.json').exists())

    def test_statistics_committed_conditions_failure_is_partially_resumable(self):
        stats=self.statistics()
        def interrupted(*args):raise RuntimeError('synthetic condition build interrupted')
        with self.assertRaises(RuntimeError):self.conditions(stats,builder=interrupted)
        failed=Path(self.cache.receipt()['operations'][-1]['attempt'])
        cold=reuse.PreparationReuse(self.root)
        same=self.statistics(cache=cold)
        self.conditions(same,cache=cold)
        self.assertEqual(self.counts['statistics'],1);self.assertEqual(self.counts['conditions'],1)
        self.assertTrue((failed/'failed.json').exists())

    def test_unreceipted_orphan_payload_never_trusted(self):
        publish=reuse._publish
        def fail_receipt(path,value):
            if path.name=='receipt.json':raise RuntimeError('interrupted after payload only')
            return publish(path,value)
        with mock.patch.object(reuse,'_publish',side_effect=fail_receipt),self.assertRaises(RuntimeError):self.statistics()
        attempt=Path(self.cache.receipt()['operations'][-1]['attempt']);orphan=(attempt/'payload.json').read_bytes()
        self.assertFalse((attempt/'receipt.json').exists())
        self.statistics(cache=reuse.PreparationReuse(self.root))
        self.assertEqual(self.counts['statistics'],2)
        self.assertEqual((attempt/'payload.json').read_bytes(),orphan)

    def test_lock_timeout_does_not_steal_or_delete_lock(self):
        path=self.root/'.lock'
        with path.open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with self.assertRaises(TimeoutError):self.statistics(cache=reuse.PreparationReuse(self.root,lock_timeout=.01))
            fcntl.flock(lock,fcntl.LOCK_UN)
        self.statistics(cache=reuse.PreparationReuse(self.root,lock_timeout=0))
        self.assertTrue(path.exists())

    @unittest.skipUnless(hasattr(os,'fork'),'POSIX process-death recovery test')
    def test_process_death_leaves_started_but_no_permanent_lock(self):
        sys.stdout.flush();sys.stderr.flush()
        pid=os.fork()
        if pid==0:
            def terminate(*args,**kwargs):os._exit(23)
            try:self.statistics(cache=reuse.PreparationReuse(self.root),builder=terminate)
            except BaseException:os._exit(99)
            os._exit(98)
        _,status=os.waitpid(pid,0)
        self.assertEqual(os.waitstatus_to_exitcode(status),23)
        started=list(self.root.glob('*/statistics/attempt-*/started.json'))
        self.assertEqual(len(started),1)
        self.statistics(cache=reuse.PreparationReuse(self.root,lock_timeout=0))
        self.assertTrue(started[0].exists())

    def test_publish_refuses_overwrite_and_cache_symlink(self):
        path=self.directory/'immutable.json';reuse._publish(path,{'first':True})
        with self.assertRaises(FileExistsError):reuse._publish(path,{'first':False})
        self.assertEqual(json.loads(path.read_text()),{'first':True})
        link=self.directory/'bad_root';link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):reuse.PreparationReuse(link)


if __name__=='__main__':
    try:unittest.main(verbosity=2)
    finally:
        print('CPU peak RSS MiB:',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              '| torch imported:', 'torch' in sys.modules,'| Isaac imported:', 'isaacsim' in sys.modules)
