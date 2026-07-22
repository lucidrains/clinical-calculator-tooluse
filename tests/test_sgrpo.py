import torch
from torch import nn
from torch.nn import Module

from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

from clinical_calculator_tooluse.sgrpo import (
    generate_sgrpo_rollouts,
    compute_sgrpo_advantages,
    SGRPOLoss
)

# helpers

def exists(val):
    return val is not None

# mocks

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0

    def encode(self, text, return_tensors='pt'):
        return torch.tensor([[1, 2, 3]])

class HFCompatibleModel(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def generate(
        self,
        input_ids,
        max_new_tokens,
        do_sample = True,
        pad_token_id = None,
        return_dict_in_generate = False
    ):
        out = self.model.generate(
            input_ids,
            seq_len = max_new_tokens,
            eos_token = pad_token_id
        )

        if not return_dict_in_generate:
            return out

        class Output: pass
        ret = Output()
        ret.sequences = out
        return ret

# test

def test_sgrpo():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 256,
            depth = 2,
            heads = 4
        )
    )

    model = AutoregressiveWrapper(model)
    hf_model = HFCompatibleModel(model)
    tokenizer = MockTokenizer()

    prompt = torch.randint(0, 20000, (1, 10))

    rollouts = generate_sgrpo_rollouts(
        hf_model,
        tokenizer,
        prompt,
        m = 2,
        max_new_tokens = 20
    )

    assert len(rollouts) == 3
    assert rollouts[-1]['is_full']
    assert all(['position' in r and 'sequence' in r for r in rollouts])

    is_correct = [True, False, True]
    advantages = compute_sgrpo_advantages(is_correct)

    assert advantages.shape == (3,)
    assert torch.allclose(advantages.sum(), torch.tensor(0.))

    loss_fn = SGRPOLoss(beta = 0.04, epsilon = 0.2)

    policy_logprobs = torch.randn(3, requires_grad = True)
    ref_logprobs = torch.randn(3)

    loss = loss_fn(policy_logprobs, ref_logprobs, advantages)

    assert loss.ndim == 0

    loss.backward()
    assert exists(policy_logprobs.grad)

def test_sgrpo_loss_with_mask():
    loss_fn = SGRPOLoss(beta = 0.04, epsilon = 0.2)
    policy_logprobs = torch.randn(2, 5, requires_grad = True)
    ref_logprobs = torch.randn(2, 5)
    advantages = torch.randn(2, 5)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype = torch.bool)

    loss = loss_fn(policy_logprobs, ref_logprobs, advantages, mask = mask)
    assert loss.ndim == 0
    loss.backward()
    assert exists(policy_logprobs.grad)

def test_attention_transformer():
    from clinical_calculator_tooluse.transformer import Attention
    attn = Attention(dim = 256, heads = 4, causal = True, num_null_kv = 2)
    x = torch.randn(2, 16, 256)
    out = attn(x)
    assert out.shape == (2, 16, 256)
