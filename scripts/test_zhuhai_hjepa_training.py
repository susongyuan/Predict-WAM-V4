#!/usr/bin/env python3
"""CPU-only sidecar wiring checks; mocked plans and tiny temporary metadata only."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from harness import zhuhai_hjepa_training as hj, zhuhai_training as base
from harness.zhuhai_preparation_reuse import PreparationReuse
from scripts.test_zhuhai_hjepa_dataset import Episode, ForbiddenStore
from scripts.test_zhuhai_preparation_reuse import config, source


def split_rows():
    splits = {hj.TASK: {split: [] for split in ('train', 'val', 'heldout')}}
    for family in range(100):
        split = 'train' if family < 80 else 'val' if family < 90 else 'heldout'
        for domain in ('source', 'randomized'):
            name = f'{family}-{domain}'
            splits[hj.TASK][split].append(dict(
                split=split, family_id=str(family), domain=domain,
                path=f'/cpu-only-hjepa-fixtures/{name}.hdf5',
                sha256=hashlib.sha256(name.encode()).hexdigest(),
                seed=1000 + 2*family + (domain == 'randomized'), **base.protocol_binding()))
    return splits, {hj.TASK: dict(quotas=dict(source=100, randomized=100))}


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='hjepa-sidecar-cpu-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = config()
        self.feature = dict(encoder_sha256='a'*64, preprocess_sha256='b'*64)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertNotIn('isaacsim', sys.modules)

    def plan(self, *, variant='combined', steps=50000):
        args = SimpleNamespace(
            mode='plan', execute=False, variant=variant, steps=steps, batch_size=4,
            save_every=5000, seed=20261009, source_count=100, randomized_count=100,
            cache_budget_bytes=10**12, cache_ram_budget_bytes=2*hj.GIB,
            minimum_free_bytes=100*hj.GIB, max_updates=None,
            initialization=str(hj.INITIALIZATION), initialization_sha256=hj.INITIALIZATION_SHA,
            tasks='not-read.json', out=self.root/'run', cache_root=self.root/'features',
            preparation_root=self.root/'preparation')
        cfg = copy.deepcopy(self.cfg)
        cfg['model']['vision_encoder']['checkpoint_path'] = '/cpu-only-encoder'
        cfg['training']['auxiliary_loss'] = dict(future=.5, gate_kl=0., phase_ce=0., log_dist=0.)
        checkpoint = dict(contract=dict(config=cfg))
        hashes = {str(ROOT/name): 'c'*64 for name in (*hj.NEW_SOURCES, *hj.BASE_SOURCES)}
        # No real manifests, weights, encoder, fitting, cache producer or GPU owner.
        with patch.object(hj, 'admit_tasks', return_value=({}, {})), \
             patch.object(hj, 'source_hashes', return_value=(hashes, [])), \
             patch.object(base, 'checkpoint_header', return_value=(checkpoint, {})), \
             patch.object(hj, 'read_json', return_value=dict(files={'model.safetensors': {'sha256': 'd'*64}})):
            result, _ = hj.make_plan(args)
        self.assertEqual(result['status'], 'cpu_preflight_completed')
        self.assertFalse(result['cuda_initialized'])
        self.assertEqual(cfg['training']['auxiliary_loss']['future'], .5)
        return result['contract']

    def test_variant_loss_weights_match_wrapper_contract(self):
        from harness.wam_hjepa import variant_settings
        for variant in hj.VARIANTS:
            with self.subTest(variant=variant):
                contract = self.plan(variant=variant)
                settings = variant_settings(variant)
                expected = dict(future=settings['future32_weight'], future8=settings['future8_weight'],
                                idm=settings['idm_weight'], gate_kl=0., phase_ce=0., log_dist=0.)
                self.assertEqual(contract['config']['training']['auxiliary_loss'], expected)
                self.assertFalse(contract['hjepa']['labels']['future_is_policy_input'])
                self.assertFalse(contract['hjepa']['labels']['causal_prefix_claim'])
                self.assertFalse(contract['automatic_policy_evaluation'])

    def test_test_milestones_follow_requested_budget_without_launching(self):
        for steps, expected in ((50000, [40000, 45000, 50000]), (100000, [90000, 95000, 100000]),
                                (12000, [2000, 7000, 12000]), (5000, [5000])):
            with self.subTest(steps=steps):
                contract = self.plan(steps=steps)
                self.assertEqual(contract['hjepa']['suggested_test_milestones'], expected)
                self.assertFalse(contract['hjepa']['automatic_evaluation'])

    def test_exact_source_family_split_and_no_numeric_string_alias(self):
        splits, sources = split_rows()
        self.assertIs(hj.validate_splits(splits, sources), splits)
        # grouped_split canonicalizes family identifiers to str; admission must agree.
        for row in splits[hj.TASK]['train'][:2]:
            row['family_id'] = 80
        with self.assertRaisesRegex(ValueError, 'family.*split'):
            hj.validate_splits(splits, sources)

    def test_all_split_metadata_checked_before_constructing_any_dataset(self):
        splits, _ = split_rows()
        splits[hj.TASK]['val'][0]['family_id'] = splits[hj.TASK]['train'][0]['family_id']
        with patch('harness.zhuhai_hjepa_dataset.HybridHjepaTaskDataset',
                   side_effect=AssertionError('opened data before full split preflight')):
            with self.assertRaisesRegex(ValueError, 'family.*split'):
                hj.build_datasets(splits, ForbiddenStore(), self.cfg, variant='combined')

    def test_dataset_wiring_preserves_order_anchors_and_auxiliary_flags(self):
        episodes = [Episode(f'{split}-{i}', n=4+i, marker=1+3*s+i, split=split)
                    for s, split in enumerate(('train', 'val', 'heldout')) for i in range(2)]
        for episode in episodes:
            episode.row.update(base.protocol_binding())
        splits = {hj.TASK: {split: [ep.row for ep in episodes if ep.row['split'] == split]
                            for split in ('train', 'val', 'heldout')}}
        by_path = {str(ep.path): ep for ep in episodes}
        for variant in hj.VARIANTS:
            with self.subTest(variant=variant), \
                 patch('harness.zhuhai_dataset.SourceEpisode', side_effect=lambda row: by_path[row['path']]), \
                 patch.object(base, 'validate_episode_binding'):
                datasets = hj.build_datasets(splits, ForbiddenStore(), self.cfg, variant=variant)
                self.assertEqual(datasets['heldout'], {})
                for split, rows in splits[hj.TASK].items():
                    if split == 'heldout': continue
                    ds = datasets[split][hj.TASK]
                    self.assertEqual([str(ep.path) for ep in ds.episodes], [r['path'] for r in rows])
                    self.assertEqual((len(ds), ds.ends, ds.chunk, ds.future_offset), (9, [4, 9], 32, 32))
                    self.assertEqual(ds.multi_horizon, variant in ('multiscale', 'combined'))
                    self.assertEqual(ds.use_idm, variant in ('idm', 'combined'))

    def test_preparation_rejects_nontrain_before_parent_identity_reads(self):
        bad = {hj.TASK: SimpleNamespace(task=hj.TASK, episodes=[
            SimpleNamespace(row=dict(split='train')), SimpleNamespace(row=dict(split='heldout'))])}
        for kind in ('statistics', 'idm_statistics', 'conditions'):
            with self.subTest(kind=kind):
                cache = hj.preparation_cache(self.root/kind, dict(labels=hj.LABEL_CONTRACT))
                with patch.object(PreparationReuse, '_identity',
                                  side_effect=AssertionError('parent may read HDF before TRAIN preflight')):
                    with self.assertRaisesRegex(ValueError, 'TRAIN'):
                        if kind == 'conditions':
                            stats = dict(normalization=dict(provenance=dict(feature_identity=self.feature)))
                            cache.conditions(dict(train=bad, val={}, heldout={}), self.cfg, stats, builder=None)
                        else:
                            extra = dict(builder=None) if kind == 'statistics' else {}
                            getattr(cache, kind)(bad, self.cfg, feature_identity=self.feature, **extra)

    def test_idm_preparation_small_train_only_cache_roundtrip(self):
        ds = source(self.root, self.cfg, hj.TASK, 101)
        original = ds.episodes[0]
        ep = Episode(n=len(original.state), state=original.state, action=original.action)
        vars(ep).update(vars(original))
        ds.task, ds.episodes, ds.store = hj.TASK, [ep], ForbiddenStore()
        train = {hj.TASK: ds}
        binding = dict(labels=copy.deepcopy(hj.LABEL_CONTRACT))
        cache = hj.preparation_cache(self.root/'reuse', binding)
        result = cache.idm_statistics(train, self.cfg, feature_identity=self.feature)
        self.assertEqual((result['valid_windows'], result['valid_label_vectors']), (3, 12))
        self.assertEqual(result['fit_split'], 'train')
        with patch('harness.zhuhai_hjepa_dataset.fit_idm_statistics',
                   side_effect=AssertionError('cache hit must not refit')):
            reused = hj.preparation_cache(self.root/'reuse', binding)
            self.assertEqual(result, reused.idm_statistics(train, self.cfg, feature_identity=self.feature))
        self.assertEqual(reused.receipt()['operations'][-1]['outcome'], 'hit')
        self.assertLess(sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()), 1024**2)


class EntryAndLoadingTests(unittest.TestCase):
    def test_default_30k_and_explicit_execution(self):
        from scripts import train_zhuhai_wam_hjepa as train
        args=train.parser().parse_args(['--out','outputs/hjepa_cpu_test_only'])
        self.assertEqual((args.steps,args.batch_size,args.save_every,args.variant),(30000,4,5000,'combined'))
        self.assertEqual(args.mode,'plan');self.assertFalse(args.execute)
        hj.validate_request(args)
        with self.assertRaises(ValueError):
            hj.validate_request(train.parser().parse_args(['--out','outputs/hjepa_cpu_test_only','--execute']))

    def test_writer_lock_does_not_reject_its_own_empty_output(self):
        from scripts import train_zhuhai_wam_hjepa as train
        with tempfile.TemporaryDirectory() as directory, patch.object(train,'reserve'), \
             patch.object(hj,'make_plan',return_value=({'status':'cpu_preflight_completed'},{})), \
             patch.object(train,'execute',return_value=0) as execute:
            out=Path(directory)/'run'
            self.assertEqual(train.main(['--out',str(out),'--mode','train','--execute']),0)
            execute.assert_called_once()
            (out/'preserved.json').write_text('{}')
            with self.assertRaises(FileExistsError):train.main(['--out',str(out),'--mode','train','--execute'])
            self.assertTrue((out/'preserved.json').exists())

    def test_aux_wrapper_and_policy_loader_roundtrip(self):
        from scripts.test_wam_hjepa import fixture
        import numpy as np
        torch.set_num_threads(1)
        idm={'scale16':[.1]*16}
        def build(cfg,stats,**kwargs):
            wrapper,_=fixture()
            if kwargs.get('state_dict') is not None:
                self.assertFalse(any(k.startswith(hj.AUX_PREFIX) for k in kwargs['state_dict']))
                wrapper.action_model.load_state_dict(kwargs['state_dict'],strict=True)
            return wrapper,{}
        with patch.object(base,'build_wrapper',side_effect=build):
            wrapper,_=hj.build_wrapper({}, {}, variant='combined',idm_statistics=idm,initialization={})
            state=wrapper.action_model.state_dict()
            loaded,_=hj.build_wrapper({}, {}, variant='combined',idm_statistics=idm,state_dict=state)
            self.assertTrue(all(torch.equal(v,loaded.action_model.state_dict()[k]) for k,v in state.items()))
            bad=copy.deepcopy(state);bad['hjepa_aux.idm_scale']*=2
            with self.assertRaisesRegex(ValueError,'IDM scale'):
                hj.build_wrapper({}, {}, variant='combined',idm_statistics=idm,state_dict=bad)
            contract=dict(config={},statistics={},idm_statistics=idm,hjepa={'variant':'combined'},
                          conditions={hj.TASK:[0.]*8},source_files={str(ROOT/'harness/wam_hjepa.py'):hj.sha256_file(ROOT/'harness/wam_hjepa.py')})
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'checkpoint_step_020000.pt'
                value=dict(format=hj.FORMAT,model_only=True,step=20000,model_state_dict=state,
                           contract=contract,contract_sha256=hj.digest(contract))
                torch.save(value,path)
                hj.write_json(path.with_name(path.name+'.sha256.json'),dict(model_only=True,step=20000,size=path.stat().st_size,sha256=hj.sha256_file(path)))
                with patch.object(hj,'validate_training_contract',side_effect=lambda c:c):
                    policy,condition,actual=hj.load_policy(path,hj.TASK,device='cpu')
                    self.assertEqual(actual,contract);self.assertEqual(condition.dtype,np.float32)
                    self.assertEqual(policy.settings['variant'],'combined');self.assertFalse(policy.training)
                    with self.assertRaises(ValueError):hj.load_policy(path,'zhuhai_task1',device='cpu')
        self.assertFalse(torch.cuda.is_initialized())

    def test_online_worker_imports_only_new_loader(self):
        import ast
        tree=ast.parse((ROOT/'scripts/eval_zhuhai_wam_hjepa.py').read_text())
        imports=[n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module=='harness']
        self.assertFalse(any(a.name=='zhuhai_training' for n in imports for a in n.names))
        self.assertTrue(any(a.name=='zhuhai_hjepa_training' for n in imports for a in n.names))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
