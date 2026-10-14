"""Synthetic admission regression for 13 preserved + 87 corrected DR."""
import json,tempfile,unittest
import builtins,symtable,sys
from unittest.mock import patch
from pathlib import Path
from copy import deepcopy
from scripts.test_zhuhai_task2_first import ShardTests as OldShardTests
from scripts.test_zhuhai_pipeline import fixture_row
from scripts import collect_isaac_zhuhai_recycle as old
from scripts import collect_isaac_zhuhai_tasks12_repair as c

class RepairAdmissionTests(unittest.TestCase):
    def setUp(self):
        fixture=OldShardTests('test_uneven_three_shards_exact_100_preserve_source_and_rows');fixture.setUp();self.addCleanup(fixture.fixture.tmp.cleanup)
        self.root=fixture.root;self.source=fixture.fixture.source
        # Extend only these temporary synthetic fixtures with fields present in
        # real source/DR contracts; no production evidence is changed.
        source=json.loads(self.source.read_text());source['contract'].update(task='zhuhai_task2',source_probe=False,allow_unvalidated_probe=False,collection_scope='production')
        for row in source['attempts']:
            row['family_id']=f'life:zhuhai_task2:source:{row["seed"]}';row['metadata'].update(task='zhuhai_task2',family_id=row['family_id'],contract=deepcopy(source['contract']))
        self.source.write_text(json.dumps(source))
        fixture.contract.update(task='zhuhai_task2',source_probe=False,allow_unvalidated_probe=False,collection_scope='production')
        fixture.contract['source_manifest']['sha256']=c.sha256(self.source)
        refs=[];rows=[]
        for i,path in enumerate(fixture.components):
            data=json.loads(path.read_text());data['contract']=deepcopy(fixture.contract);data['attempts']=data['attempts'][:[4,5,4][i]];data['success_count']=len(data['attempts']);data['status']='stopped_consecutive_failures'
            for row in data['attempts']:
                row['family_id']=f'life:zhuhai_task2:source:{row["source_parent"]}';row['metadata'].update(task='zhuhai_task2',family_id=row['family_id'],contract=deepcopy(data['contract']))
            path.write_text(json.dumps(data));rows+=data['attempts'];refs.append(dict(path=str(path),sha256=c.sha256(path),successful_episodes=len(data['attempts'])))
        self.prior=self.root/'prior.json';self.prior.write_text(json.dumps(dict(components=refs,attempts=sorted(rows,key=lambda x:x['seed']),successful_episodes=13,old_collector=dict(path=str(Path(old.__file__).resolve()),sha256=c.sha256(old.__file__)))))
        self.contract=deepcopy(fixture.contract);self.contract.update(seed_start=20000,max_consecutive_failures=3,collection_implementation=dict(path=str(Path(c.__file__).resolve()),sha256=c.sha256(c.__file__)),prior_inventory=dict(path=str(self.prior),sha256=c.sha256(self.prior)),expert_variant=dict(path=str(c.ROOT/'harness/isaac_zhuhai_task2_expert_repair_v3.py'),sha256=c.sha256(c.ROOT/'harness/isaac_zhuhai_task2_expert_repair_v3.py')))
        self.contract['collection_sharding']['shard_count']=4
        self.contract['max_attempts']=1000
        self.parts=[]
        for index in range(4):
            parents=c.residual_families(self.contract,index,4);directory=self.root/f'new_{index}';directory.mkdir()
            rows=[fixture_row(directory,self.contract,20000+index*1000+j,parent=parent['seed']) for j,parent in enumerate(parents)]
            for row in rows:row['metadata']['task']='zhuhai_task2'
            manifest=directory/'manifest.json';manifest.write_text(json.dumps(dict(contract=self.contract,status='complete',success_count=len(rows),requested_success_count=len(rows),attempts=rows)))
            self.parts.append(manifest)

    def test_13_original_plus_87_new_exactly_100_no_evidence_relabel(self):
        before=self.prior.read_bytes();expected=c.prior_rows(self.prior)
        self.assertEqual(sum(len(c.validate_manifest(path,task='zhuhai_task2',mode='randomized',shard_index=i,shard_count=4)) for i,path in enumerate(self.parts)),87)
        aggregate=c.aggregate_shards(self.root/'aggregate.json',self.parts,task='zhuhai_task2');admitted=c.validate_manifest(aggregate,task='zhuhai_task2',mode='randomized')
        self.assertEqual(len(admitted),100);self.assertEqual(self.prior.read_bytes(),before)
        self.assertEqual([x for x in admitted if x['seed']<20000],expected)

    def test_changed_old_component_rejected(self):
        ref=json.loads(self.prior.read_text())['components'][0];Path(ref['path']).write_bytes(Path(ref['path']).read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Prior component changed'):c.prior_rows(self.prior)

    def test_incomplete_residual_never_admitted(self):
        path=self.parts[0];data=json.loads(path.read_text());data['attempts'].pop();data['success_count']-=1;path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'Incomplete residual'):c.validate_manifest(path,task='zhuhai_task2',mode='randomized',shard_index=0,shard_count=4)

    def test_wrong_residual_family_rejected(self):
        path=self.parts[0];data=json.loads(path.read_text());row=data['attempts'][0];parent=c.prior_rows(self.prior)[0]
        row.update(family_id=parent['family_id'],source_parent=parent['source_parent']);row['metadata'].update(family_id=parent['family_id'],parent_source_seed=parent['source_parent']);path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'family coverage'):c.validate_manifest(path,task='zhuhai_task2',mode='randomized',shard_index=0,shard_count=4)

    def test_recycle_boundary_not_training_complete(self):
        path=self.parts[0];data=json.loads(path.read_text());data.update(status='worker_recycle_boundary',attempts=data['attempts'][:3],success_count=3);path.write_text(json.dumps(data))
        self.assertFalse(c.validate_recycle_boundary(path,task='zhuhai_task2')['quota_complete'])
        with self.assertRaises(ValueError):c.validate_manifest(path,task='zhuhai_task2',mode='randomized',shard_index=0,shard_count=4)

    def test_header_rejects_protocol_drift(self):
        mutations=[('camera_count',2),('minimum_free_bytes',0),('allow_unvalidated_probe',True),
            ('source_probe',None),('collection_scope','expert_repair_probe_not_production'),
            ('max_attempts',2000),('seed_start',-1),('max_consecutive_failures',2),
            ('source_quota_transition',dict(from_quota=c.source_protocol.PRODUCTION_QUOTAS,to_quota=c.PRODUCTION_QUOTAS,source_relabelled=True))]
        for field,value in mutations:
            with self.subTest(field=field):
                modified=deepcopy(self.contract);modified[field]=value
                with self.assertRaises(ValueError):c.validate_header(modified,'zhuhai_task2')
        for field,value in [('shard_count',True),('shard_count',5),('family_assignment','arbitrary'),
                ('global_randomized_count',87),('randomized_per_family',2),('disjoint_seed_stride',2000)]:
            with self.subTest(sharding=field):
                modified=deepcopy(self.contract);modified['collection_sharding'][field]=value
                with self.assertRaisesRegex(ValueError,'sharding'):c.validate_header(modified,'zhuhai_task2')

    def recycle_fixture(self):
        path=self.parts[0];data=json.loads(path.read_text());data.update(status='worker_recycle_boundary',attempts=data['attempts'][:3],success_count=3)
        return path,data

    def publish_recycle(self,path,data):
        for row in data['attempts']:row['metadata']['contract']=deepcopy(data['contract'])
        path.write_text(json.dumps(data))

    def test_recycle_verifies_prior_inventory_and_expert(self):
        for field in ('prior_inventory','expert_variant'):
            with self.subTest(binding=field):
                path,data=self.recycle_fixture();data['contract'][field]['sha256']='0'*64
                self.publish_recycle(path,data)
                with self.assertRaises(ValueError):c.validate_recycle_boundary(path,task='zhuhai_task2')
                # Restore the component fixture for the next mutation.
                data['contract']=deepcopy(self.contract);self.publish_recycle(path,data)

    def test_recycle_cannot_import_old_family_or_other_shard(self):
        path,data=self.recycle_fixture();prior=c.prior_rows(self.prior)[0];row=data['attempts'][0]
        row.update(family_id=prior['family_id'],source_parent=prior['source_parent']);row['metadata'].update(family_id=prior['family_id'],parent_source_seed=prior['source_parent'])
        self.publish_recycle(path,data)
        with self.assertRaisesRegex(ValueError,'residual family'):c.validate_recycle_boundary(path,task='zhuhai_task2')
        data['attempts'][0]=json.loads(self.parts[1].read_text())['attempts'][0];self.publish_recycle(path,data)
        with self.assertRaisesRegex(ValueError,'shard seed range'):c.validate_recycle_boundary(path,task='zhuhai_task2')

    def test_recycle_cannot_redeclare_local_quota(self):
        path,data=self.recycle_fixture();data['requested_success_count']+=1;self.publish_recycle(path,data)
        with self.assertRaisesRegex(ValueError,'residual quota'):c.validate_recycle_boundary(path,task='zhuhai_task2')

    def test_recycle_checks_failed_attempt_family_and_no_running_attempt(self):
        path,data=self.recycle_fixture();row=data['attempts'][0];row['status']='failed';data['success_count']=2;row['family_id']='unregistered'
        row['metadata']['family_id']=row['family_id'];self.publish_recycle(path,data)
        with self.assertRaisesRegex(ValueError,'residual family'):c.validate_recycle_boundary(path,task='zhuhai_task2')
        row['status']='running';self.publish_recycle(path,data)
        with self.assertRaisesRegex(ValueError,'between physical attempts'):c.validate_recycle_boundary(path,task='zhuhai_task2')

    def test_aggregate_rejects_divergent_component_contract(self):
        path=self.parts[1];data=json.loads(path.read_text());data['contract']['unregistered_protocol_change']='divergent'
        for row in data['attempts']:row['metadata']['contract']=deepcopy(data['contract'])
        path.write_text(json.dumps(data))
        combined=c.prior_rows(self.prior)
        for component in self.parts:combined.extend(json.loads(component.read_text())['attempts'])
        aggregate=self.root/'divergent_aggregate.json';aggregate.write_text(json.dumps(dict(contract=self.contract,status='complete',requested_success_count=100,success_count=100,
            attempts=sorted(combined,key=lambda r:r['seed']),shard_manifests=[dict(index=i,path=str(part),sha256=c.sha256(part)) for i,part in enumerate(self.parts)])))
        with self.assertRaisesRegex(ValueError,'component/aggregate contracts differ'):c.validate_manifest(aggregate,task='zhuhai_task2',mode='randomized')

    def cli_arguments(self,out,index=2):
        return [c.__file__,'--task','zhuhai_task2','--mode','randomized','--count','100','--seed-start','20000',
            '--shard-index',str(index),'--shard-count','4','--out',str(out),'--source-manifest',str(self.source),
            '--prior-inventory',str(self.prior),'--gpu-lock-owner','cpu-main-fixture']

    def test_main_plan_only_preserves_global_quota_and_allocates_residual_families(self):
        out=self.root/'cli_plan';argv=self.cli_arguments(out)+['--plan-only']
        before=self.source.read_bytes();expected=c.residual_families(self.contract,2,4)
        with patch.object(sys,'argv',argv),patch.object(c,'contract',return_value=deepcopy(self.contract)),patch.object(c,'collect') as collect,patch.object(c,'verify_owner') as verify:
            self.assertEqual(c.main(),0);collect.assert_not_called();verify.assert_not_called()
        plan=json.loads((out/'plan.json').read_text())
        self.assertEqual(plan['global_requested_count'],100);self.assertEqual(plan['local_requested_count'],len(expected))
        self.assertEqual(plan['local_seed_start'],22000);self.assertEqual(plan['local_max_attempts'],10*len(expected))
        self.assertEqual(self.source.read_bytes(),before)

    def test_main_reaches_collect_with_exact_residual_args_without_starting_Isaac(self):
        out=self.root/'cli_start';parents=c.residual_families(self.contract,2,4);contract=deepcopy(self.contract)
        contract['robot_binding']['assets_ready']=True;contract['runtime_interface']={'ready':True}
        with patch.object(sys,'argv',self.cli_arguments(out)),patch.object(c,'contract',return_value=contract),patch.object(c,'residual_families',return_value=parents) as allocate,patch.object(c,'collect',return_value=75) as collect,patch.object(c,'verify_owner') as verify:
            self.assertEqual(c.main(),75)
        args,observed_contract,observed_parents=collect.call_args.args
        self.assertEqual(args.count,len(parents));self.assertEqual(args.seed_start,22000);self.assertEqual(args.max_attempts,10*len(parents))
        self.assertEqual(observed_contract['requested_success_count'],100);self.assertEqual(observed_parents,parents)
        verify.assert_called_once_with('cpu-main-fixture');self.assertEqual(allocate.call_count,3)

    def test_module_has_no_unresolved_global_names(self):
        declared=set(vars(c))|set(vars(builtins));missing=set()
        def inspect(scope):
            for symbol in scope.get_symbols():
                if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in declared:missing.add(symbol.get_name())
            for child in scope.get_children():inspect(child)
        inspect(symtable.symtable(Path(c.__file__).read_text(),c.__file__,'exec'))
        self.assertEqual(missing,set())

if __name__=='__main__':unittest.main()
