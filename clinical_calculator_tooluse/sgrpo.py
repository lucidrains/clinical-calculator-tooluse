import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange

# constants

DEFAULT_EARLY_EXIT_PROMPT = "Time is limited, stop thinking and start answering.\n</think>\n\n"

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# loss

class SGRPOLoss(Module):
    def __init__(
        self,
        *,
        beta = 0.04,
        epsilon = 0.2
    ):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(
        self,
        policy_logprobs,
        ref_logprobs,
        advantages,
        mask = None
    ):
        ratio = torch.exp(policy_logprobs - ref_logprobs)
        clipped_ratio = ratio.clamp(1. - self.epsilon, 1. + self.epsilon)

        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages

        surrogate = torch.min(surrogate1, surrogate2)
        kl_div = torch.exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1.

        if exists(mask):
            policy_loss = -(surrogate * mask).sum() / mask.sum().clamp(min = 1e-8)
            kl_loss = (kl_div * mask).sum() / mask.sum().clamp(min = 1e-8) * self.beta
        else:
            policy_loss = -surrogate.mean()
            kl_loss = kl_div.mean() * self.beta

        return policy_loss + kl_loss

# advantages

def compute_sgrpo_advantages(is_correct: list[bool]):
    n_right = 0
    rewards = []

    for correct in is_correct:
        n_right += int(correct)
        rewards.append(1. / (2 ** (n_right - 1)) if correct else 0.)

    rewards = torch.tensor(rewards, dtype = torch.float32)
    return rewards - rewards.mean()

# rollouts

@torch.no_grad()
def generate_sgrpo_rollouts(
    model,
    tokenizer,
    prompt_ids,
    *,
    m = 3,
    early_exit_prompt = DEFAULT_EARLY_EXIT_PROMPT,
    max_new_tokens = 1024
):
    model.eval()

    if prompt_ids.ndim == 1:
        prompt_ids = rearrange(prompt_ids, 'n -> 1 n')

    prompt_len = prompt_ids.shape[-1]

    outputs = model.generate(
        prompt_ids,
        max_new_tokens = max_new_tokens,
        do_sample = True,
        return_dict_in_generate = True,
        pad_token_id = tokenizer.eos_token_id
    )

    full_seq = outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs[0]
    gen_seq = full_seq[prompt_len:]
    seq_len = len(gen_seq)

    positions = torch.randperm(seq_len - 1)[:m] + 1 if seq_len > m else torch.arange(1, seq_len)
    positions, _ = positions.sort()

    exit_prompt_ids = tokenizer.encode(early_exit_prompt, return_tensors = 'pt').to(prompt_ids.device)

    rollouts = []

    for pos in positions:
        truncated_seq = rearrange(full_seq[:prompt_len + pos], 'n -> 1 n')
        early_exit_seq = torch.cat((truncated_seq, exit_prompt_ids), dim = -1)

        branch_out = model.generate(
            early_exit_seq,
            max_new_tokens = max_new_tokens // 2,
            do_sample = True,
            pad_token_id = tokenizer.eos_token_id
        )

        branch_seq = branch_out.sequences[0] if hasattr(branch_out, 'sequences') else branch_out[0]

        rollouts.append(dict(
            sequence = branch_seq,
            position = pos.item(),
            is_full = False
        ))

    rollouts.append(dict(
        sequence = full_seq,
        position = seq_len,
        is_full = True
    ))

    return rollouts
