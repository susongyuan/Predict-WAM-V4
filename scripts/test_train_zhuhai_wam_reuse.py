#!/usr/bin/env python3
"""CPU-only adapter wiring tests; no model, CUDA, data fitting or live file edits."""
from pathlib import Path
import hashlib,json,sys,tempfile,types,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_zhuhai_wam_reuse import ReuseHooks,validate_recovery_source,write_receipt

class Cache:
    def __init__(self):self.calls=[]
    def statistics(self,ds,cfg,**kwargs):
        self.calls.append(('statistics',ds,cfg));kwargs['check']()
        return kwargs['builder'](ds,cfg,feature_identity=kwargs['feature_identity'],check=kwargs['check'])
    def conditions(self,ds,cfg,stats,**kwargs):
        self.calls.append(('conditions',ds,cfg));kwargs['check']()
        return kwargs['builder'](ds,cfg,stats)

class AdapterTests(unittest.TestCase):
    def module(self):
        return types.SimpleNamespace(config_for=lambda **kw:{'training':kw},
            fit_train_statistics=lambda ds,cfg,**kw:{'fit':ds,'feature':kw['feature_identity']},
            install_conditions=lambda ds,cfg,stats:{'condition':ds,'statistics':stats})
    def test_original_algorithm_delegation_and_provenance(self):
        module=self.module();old=dict(vars(module));cache=Cache();check=[]
        with ReuseHooks(module,cache,metadata={'format':'candidate','sources_sha256':{'p':'sha'}},owner_check=lambda:check.append('owner')):
            cfg=module.config_for(steps=5000)
            self.assertEqual(cfg['training'],{'steps':5000});self.assertEqual(cfg['training_preparation_execution']['format'],'candidate')
            stat=module.fit_train_statistics('TRAIN',cfg,feature_identity={'encoder':'x'},check=lambda:check.append('gpu'))
            bank=module.install_conditions('ALL_SPLITS',cfg,stat)
            self.assertEqual(bank,{'condition':'ALL_SPLITS','statistics':{'fit':'TRAIN','feature':{'encoder':'x'}}})
        self.assertEqual(vars(module),old);self.assertEqual([x[0] for x in cache.calls],['statistics','conditions'])
        self.assertIn('gpu',check);self.assertIn('owner',check)
    def test_exception_restores_functions(self):
        module=self.module();old=dict(vars(module))
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with ReuseHooks(module,Cache(),metadata={}):raise RuntimeError('intentional')
        self.assertEqual(vars(module),old)
        with ReuseHooks(module,Cache(),metadata={}):pass
    def test_nested_patch_refused(self):
        module=self.module()
        with ReuseHooks(module,Cache(),metadata={}):
            with self.assertRaises(RuntimeError):
                with ReuseHooks(module,Cache(),metadata={}):pass
    def fixture(self,root):
        old=root/'old';old.mkdir();src=root/'code.py';src.write_text('original')
        sources={}
        for task in ('zhuhai_task1','zhuhai_task2'):
            manifests=[]
            for mode in ('source','randomized'):
                manifest=root/f'{task}_{mode}.json';manifest.write_text('{"CPU_fixture":true}')
                manifests.append({'path':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
            sources[task]={'quotas':{'source':100,'randomized':500},'families':100,'manifests':manifests}
        plan={'contract':{'source_files':{str(src):hashlib.sha256(src.read_bytes()).hexdigest()},'sources':sources,'config':{'dof':16}}}
        (old/'plan.json').write_text(json.dumps(plan));return old,src
    def test_prep_retry_preserves_old_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,_=self.fixture(root);before=(old/'plan.json').read_bytes();mtime=(old/'plan.json').stat().st_mtime_ns
            result=validate_recovery_source(old,root/'new')
            self.assertFalse(result['original_modified']);self.assertEqual((old/'plan.json').read_bytes(),before)
            self.assertEqual((old/'plan.json').stat().st_mtime_ns,mtime);self.assertFalse((root/'new').exists())
    def test_same_output_or_possible_optimizer_state_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,_=self.fixture(root)
            for target in (old,old/'nested'):
                with self.assertRaises(ValueError):validate_recovery_source(old,target)
            (old/'checkpoint_last.pt').write_bytes(b'preserved')
            with self.assertRaises(ValueError):validate_recovery_source(old,root/'new')
    def test_changed_code_and_nonempty_new_out_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,src=self.fixture(root);new=root/'new';new.mkdir();(new/'keep').write_text('evidence')
            with self.assertRaises(FileExistsError):validate_recovery_source(old,new)
            src.write_text('changed')
            with self.assertRaises(ValueError):validate_recovery_source(old,root/'other')
    def test_enter_failure_does_not_poison_subsequent_hooks(self):
        broken=self.module();del broken.install_conditions
        with self.assertRaises(AttributeError):
            with ReuseHooks(broken,Cache(),metadata={}):pass
        with ReuseHooks(self.module(),Cache(),metadata={}):pass
    def test_recovery_rejects_manifest_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,_=self.fixture(root)
            (root/'zhuhai_task1_source.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):validate_recovery_source(old,root/'new')
    def test_receipt_publication_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            a=write_receipt(directory,{'status':'CPU fixture'});before=a.read_bytes();b=write_receipt(directory,{'status':'another'})
            self.assertNotEqual(a,b);self.assertEqual(a.read_bytes(),before)

if __name__=='__main__':unittest.main()
