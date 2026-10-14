import copy
import unittest
import torch
from scripts.precision_activation_checkpoint import enable
from scripts.test_wam_precision_v4 import make_pair, sample
from harness.zhuhai_precision_v4_future16 import Future16Wrapper

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_actual_small_future16_loss_gradients_rng_and_inference(self):
        _, original, batch = make_pair()
        original = Future16Wrapper(original.base, original.settings['variant'])
        batch['future_feats_16']=batch.pop('future_feats_8')
        batch['future_valid_mask_16']=batch.pop('future_valid_mask_8')
        candidate=copy.deepcopy(original)
        names=enable(candidate.action_model)
        self.assertGreater(len(names),2)
        self.assertEqual(set(original.state_dict()),set(candidate.state_dict()))
        rng=torch.get_rng_state().clone()
        original.train();candidate.train()
        loss_a,info_a,_=original(batch);loss_a.backward();after_a=torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        loss_b,info_b,_=candidate(batch);loss_b.backward();after_b=torch.get_rng_state().clone()
        torch.testing.assert_close(loss_b,loss_a,rtol=0,atol=0)
        torch.testing.assert_close(after_b,after_a,rtol=0,atol=0)
        self.assertEqual(info_a,info_b)
        for (name,a),(name_b,b) in zip(original.named_parameters(),candidate.named_parameters()):
            self.assertEqual(name,name_b)
            self.assertEqual(a.grad is None,b.grad is None,name)
            if a.grad is not None: torch.testing.assert_close(b.grad,a.grad,rtol=1e-5,atol=1e-7,msg=name)
        original.eval();candidate.eval()
        torch.testing.assert_close(sample(candidate,batch),sample(original,batch),rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_nested_inputs_and_dropout_are_preserved(self):
        class MultiLayerConcatFusion(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(8,8);self.dropout=torch.nn.Dropout(.3)
            def forward(self, values): return self.dropout(self.linear(values[0]+values[1]))
        original=MultiLayerConcatFusion();candidate=copy.deepcopy(original);enable(candidate)
        inputs=[torch.randn(3,8),torch.randn(3,8)]
        rng=torch.get_rng_state().clone()
        a=original(inputs);a.sum().backward();rng_a=torch.get_rng_state().clone()
        torch.set_rng_state(rng);b=candidate(inputs);b.sum().backward()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(rng_a,torch.get_rng_state(),rtol=0,atol=0)
        torch.testing.assert_close(original.linear.weight.grad,candidate.linear.weight.grad,rtol=0,atol=0)

if __name__ == '__main__': unittest.main()
