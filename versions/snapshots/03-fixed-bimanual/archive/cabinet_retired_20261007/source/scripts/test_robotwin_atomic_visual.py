#!/usr/bin/env python3
"""CPU-only atomic interface tests; never imports torch or loads a model."""
import base64
import copy
import json
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
from harness import robotwin_atomic_visual as a
from harness import robotwin_vlm_worker as v
OUT=ROOT/'outputs/validation/robotwin_atomic_visual_train_20261006'

def frame(n=0, episode='synthetic'):
 return {'episode_id':episode,'observation_id':n,'rgb':{view:{'encoding':'rgb8_base64','shape':[240,320,3],
   'data':base64.b64encode(bytes([11,32,64])*76800).decode()} for view in v.VIEWS}}
def req(pred='drawer_open',rid='r'):
 return {'version':1,'id':rid,'predicate':pred,'current':frame(2)}
def value(): return {'state':'unknown','views':['cam_head'],'evidence':'Key region is occluded.','object_description':'not identifiable'}
class Fake:
 def __init__(self,raw=None):self.raw=raw;self.calls=[]
 def guard(self):pass
 def resources(self):return {'synthetic_only':True}
 def complete(self,prompt,images):
  self.calls.append((prompt,images));return self.raw if self.raw is not None else json.dumps(value()),{'new_tokens':10}
class Tests(unittest.TestCase):
 def test_all_static_and_scopes(self):
  backend=Fake();adapter=a.AtomicVisualVerifier(backend)
  for i,pred in enumerate(a.STATIC_PREDICATES):
   r=adapter.query(req(pred,str(i)));self.assertEqual(r['outcome'],'ok');self.assertEqual(r['status'],v.STATUS)
   self.assertEqual(r['parsed'],value());self.assertEqual(r['claim_scope'],a.SCOPES[pred]);self.assertNotIn('verified',r)
 def test_temporal_requires_previous(self):
  with self.assertRaises(v.ProtocolError):a.decode_request(req('visible_release'))
 def test_static_rejects_previous(self):
  r=req();r['previous']=frame(1)
  with self.assertRaises(v.ProtocolError):a.decode_request(r)
 def test_temporal_binding_and_sizes(self):
  r=req('visible_release');r['previous']=frame(1);backend=Fake();out=a.AtomicVisualVerifier(backend).query(r)
  self.assertEqual(out['binding']['previous']['observation_id'],1);self.assertEqual([im.size for im in backend.calls[0][1]],[(960,264),(960,264)])
 def test_previous_cross_episode_or_forward_rejected(self):
  for old in (frame(1,'other'),frame(3),frame(2)):
   r=req('visible_release');r['previous']=old
   with self.assertRaises(v.ProtocolError):a.decode_request(r)
 def test_extra_GT_path_command_never_forwarded(self):
  for key in ('gt','filename','command','instruction','prompt','success'):
   r=req();r[key]='secret'
   with self.assertRaises(v.ProtocolError):a.AtomicVisualVerifier(Fake()).query(r)
 def test_ids_not_in_prompt(self):
  r=req(rid='TRANSPORT_SECRET');r['current']['episode_id']='EPISODE_SECRET'
  prompt,images=a.build_input(a.decode_request(r));self.assertNotIn('TRANSPORT_SECRET',prompt);self.assertNotIn('EPISODE_SECRET',prompt)
 def test_no_filled_unknown_schema_examples(self):
  for p in a.PROMPTS.values():self.assertNotIn('"state":"unknown"',p);self.assertNotIn('"state": "unknown"',p)
 def test_parse_requires_description_and_strict_fields(self):
  for key in ('object_description','evidence','views','state'):
   val=value();del val[key]
   with self.assertRaises(v.ProtocolError):a.parse(json.dumps(val))
 def test_no_authority_or_bool_state(self):
  for val in ({**value(),'verified':True},{**value(),'state':True},{**value(),'actions':[0]*14}):
   with self.assertRaises(v.ProtocolError):a.parse(json.dumps(val))
 def test_parse_error_keeps_raw_not_unknown_vote(self):
  raw='{"state":"true"}';out=a.AtomicVisualVerifier(Fake(raw)).query(req())
  self.assertIsNone(out['parsed']);self.assertEqual(out['raw_text'],raw);self.assertEqual(out['outcome'],'parse_error')
 def test_duplicate_id(self):
  model=a.AtomicVisualVerifier(Fake());model.query(req())
  with self.assertRaises(v.ProtocolError):model.query(req())
 def test_same_obs_three_independent_prompts(self):
  model=a.AtomicVisualVerifier(Fake())
  for p in a.STATIC_PREDICATES:self.assertEqual(model.query(req(p,p))['outcome'],'ok')
 def test_prompt_boundaries(self):
  self.assertIn('NOT grasp retention',a.PROMPTS['object_between_gripper_fingers'])
  self.assertIn('NOT certified 3D containment',a.PROMPTS['object_visibly_inside_open_drawer'])
  self.assertIn('same identified object',a.PROMPTS['visible_release'])
 def test_cpu_only(self):self.assertNotIn('torch',sys.modules)
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);program=unittest.main(exit=False,verbosity=2);r=program.result
 (OUT/'cpu_tests.json').write_text(json.dumps({'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'gpu_used':False,'torch_imported':'torch' in sys.modules,'admission':False},indent=2)+'\n')
 raise SystemExit(0 if r.wasSuccessful() else 1)
