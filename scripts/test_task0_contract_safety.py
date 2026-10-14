#!/usr/bin/env python3
"""CPU synthetic contract regressions; not physical outcome verification."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import unittest
import numpy as np
from harness.task0_predicates import Task0Postconditions
from harness.orchestrator import (Catalog, ConditionBank, Orchestrator, CandidateProgram,
                                  Step, Contract, _FakeClosure, _FakePrecond)


class ReleaseIsolation(unittest.TestCase):
    def run_release(self, *, reset, guard=False, first=False, height=1.2):
        p = Task0Postconditions(holding_guard=guard)
        s = np.zeros(61); s[49:51] = .019; s[44] = .65
        a = np.zeros(23)
        if not first:
            p.start('place', 'place on', ('radio', 'table'), s, 0)
            p.update(s, a, 1)
            p._release_ok = True
        if reset:
            p.reset_episode()
            self.assertIsNone(p._pre_z)
            self.assertFalse(p._release_ok)
            self.assertIsNone(p.update(s, a, 2))
        s[44] = height
        p.start('place', 'place on', ('radio', 'table'), s, 0)
        self.assertFalse(p._release_ok)
        s[49:51] = .05
        p.update(s, a, 1)
        ev = p.update(s, a, 11)
        self.assertAlmostEqual(ev['detail']['release_z'], height)
        return ev

    def test_old_episode_height_cannot_certify_high_release(self):
        for guard in (False, True):
            with self.subTest(guard=guard):
                ev = self.run_release(reset=True, guard=guard)
                self.assertTrue(ev['failed']); self.assertFalse(ev['satisfied'])

    def test_old_contract_height_cannot_certify_high_release(self):
        self.assertTrue(self.run_release(reset=False)['failed'])

    def test_first_contract_initializes_pre_release_height(self):
        self.assertTrue(self.run_release(reset=False, first=True)['failed'])

    def test_current_surface_height_still_passes_proxy(self):
        self.assertTrue(self.run_release(reset=True, height=.66)['satisfied'])

    def test_grasp_height_persists_only_within_episode(self):
        p = Task0Postconditions(); s = np.zeros(61)
        p.grasp_z = .64
        p.start('new', 'place on', ('radio', 'table'), s, 10)
        self.assertEqual(p.grasp_z, .64)
        p.reset_episode(); self.assertIsNone(p.grasp_z)


class CandidateSafety(unittest.TestCase):
    def make(self, **kw):
        cat = Catalog()
        protos = {name: np.full(1024, .1, np.float32) for name in cat.patterns}
        opts = dict(catalog=cat, bank=ConditionBank(protos), closure_monitor=_FakeClosure(2),
                    precondition_monitor=_FakePrecond(1), budgets={'move to(car)':100, 'open lid(car)':20},
                    max_attempts=2, budget_factor=1, navigation_settle_steps=1,
                    manipulation_completion='postcondition', candidate_advance_kinds=('contact_candidate',))
        opts.update(kw)
        o = Orchestrator(**opts)
        o.reset([CandidateProgram('lid', [Step('move to', ('car',)), Step('open lid', ('car',))])])
        return o

    def candidate(self, t, **kw):
        return dict(satisfied=True, failed=False, verified=False, verifier_kind='contact_candidate',
                    outcome_observable=False, retry_safe=False, source='synthetic://candidate',
                    key='open lid(car)', observation_id=t, **kw)

    def drive(self, o, evidence=None):
        for t in range(80):
            d = o.tick(np.zeros(1024), t, base_velocity=np.zeros(3),
                       postcondition=self.candidate(t) if evidence is None else evidence(t))
            if o.status in ('held', 'done'):
                return d, t
        return d, t

    def test_default_candidate_holds_in_both_completion_modes(self):
        for mode in ('postcondition', 'vtt_and_postcondition'):
            with self.subTest(mode=mode):
                o = self.make(manipulation_completion=mode)
                d, t = self.drive(o)
                self.assertEqual(d.mode, 'HOLD')
                self.assertEqual(o.index, 1); self.assertEqual(len(o.completed_steps), 1)
                self.assertEqual(o.pending_hold_reason, 'candidate_requires_independent_verification')
                ends = [e for e in o.events if e['event']=='contract_end' and e['key']=='open lid(car)']
                self.assertEqual(len(ends), 1)
                self.assertIsNone(ends[0]['success']); self.assertFalse(ends[0]['verified'])
                self.assertFalse(any(a.successes or a.failures or a.support_count for arms in o.bank.arms.values() for a in arms))
                for later in range(t+1, t+10):
                    self.assertEqual(o.tick(np.zeros(1024), later, postcondition=self.candidate(later)).mode, 'HOLD')
                again = Contract(Step('open lid', ('car',)), 1, None, 20, arm_key='open lid(car)')
                self.assertFalse(o._start_contract(again, np.zeros(1024), t+20))

    def test_legacy_requires_explicit_opt_in_and_never_reports_success(self):
        o = self.make(candidate_outcome_policy='legacy_advance')
        d, _ = self.drive(o)
        self.assertEqual(d.mode, 'DONE')
        end = [e for e in o.events if e['event']=='contract_end' and e['key']=='open lid(car)'][0]
        self.assertIsNone(end['success']); self.assertTrue(end['legacy_program_advanced'])

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError): self.make(candidate_outcome_policy='advance')

    def test_stale_wrong_key_or_self_verified_candidate_cannot_pass(self):
        o = self.make()
        for patch in ({'observation_id':9}, {'key':'other'}, {'verified':True}, {'outcome_observable':True}):
            ev = {**self.candidate(10), **patch}
            self.assertIsNone(o._candidate_evidence(ev, key='open lid(car)', observation_id=10))

    def test_possible_contact_cannot_be_erased_by_later_retry_safe_record(self):
        o = self.make()
        seen = []
        def ev(t):
            if o.contract is not None and not o.contract.is_navigation: seen.append(t)
            return {**self.candidate(t), 'satisfied':False, 'retry_safe':len(seen)>1}
        d, _ = self.drive(o, ev)
        self.assertEqual(d.mode, 'HOLD')
        self.assertEqual(o.pending_hold_reason, 'irreversible_outcome_unknown')
        self.assertEqual(sum(e['event']=='contract_start' and e['key']=='open lid(car)' for e in o.events), 1)

    def test_independent_synthetic_evidence_still_completes(self):
        o = self.make()
        d, _ = self.drive(o, lambda t: dict(satisfied=True, failed=False, verified=True,
            verifier_kind='controller_postconditions', source='synthetic://independent-test-only',
            observation_id=t, key='open lid(car)'))
        self.assertEqual(d.mode, 'DONE')
        self.assertTrue(o.realized[-1]['success'])


if __name__ == '__main__':
    unittest.main()
