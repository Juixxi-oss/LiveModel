import math
import random
import numpy as np
import torch
import torch.nn as nn

from model import AdjustableDropout


def setSeed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs: int = 5,
        start_factor: float = 0.1,
        T_0: int = 15,
        T_mult: int = 2,
        eta_min: float = 1e-7,
):
    if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        raise ValueError("warmup_epochs must be a non-negative integer")
    if not 0 < start_factor <= 1:
        raise ValueError("start_factor must be in (0, 1]")
    if not isinstance(T_0, int) or T_0 < 1:
        raise ValueError("T_0 must be a positive integer")
    if not isinstance(T_mult, int) or T_mult < 1:
        raise ValueError("T_mult must be a positive integer")
    if not 0 <= eta_min <= min(group["lr"] for group in optimizer.param_groups):
        raise ValueError("eta_min must be between zero and the base learning rate")

    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    if warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def get_suplambda(
        epoch: int,
        warmup_epochs: int = 5,
        stable_epochs: int = 5,
        decay_epochs: int = 35,
        max_lambda: float = 0.1,
        min_lambda: float = 0.001
) -> float:
    if epoch < warmup_epochs:
        return max_lambda * (epoch + 1) / warmup_epochs

    elif epoch < warmup_epochs + stable_epochs:
        return max_lambda

    elif epoch < warmup_epochs + stable_epochs + decay_epochs:
        decay_ratio = (epoch - warmup_epochs - stable_epochs) / decay_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lambda + (max_lambda - min_lambda) * cosine_decay

    else:
        return min_lambda


def update_dropout(
        model: nn.Module,
        current_epoch: int,
        total_epochs: int,
        max_dropout: float = 0.3,
        min_dropout: float = 0.1
) -> float:
    new_p = min_dropout + (max_dropout - min_dropout) * (
        current_epoch / max(total_epochs, 1)
    )

    for module in model.modules():
        if isinstance(module, AdjustableDropout):
            module.set_dropout(new_p)

    return new_p


def generate_attention_1(
        grade_type: str,
        batch_size: int,
        device: torch.device
) -> torch.Tensor:
    if grade_type == 'C':
        attention_lst = [0.01, 0.97, 0.01, 0.01]

    elif grade_type == 'N':
        attention_lst = [0.01, 0.01, 0.97, 0.01]

    elif grade_type == 'P':
        attention_lst = [0.01, 0.01, 0.01, 0.97]

    elif grade_type in ['BCVA', 'SR']:
        attention_lst = [0.20, 0.15, 0.20, 0.45]

    else:
        attention_lst = [0.20, 0.15, 0.45, 0.20]

    return torch.tensor(
        attention_lst,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0).repeat(batch_size, 1)


def generate_attention_2(
        batch_size: int,
        time_steps: int,
        src_key_padding_mask: torch.Tensor,
        device: torch.device
) -> torch.Tensor:
    """Return a soft recency prior over available visit positions [B, T].

    This target supervises temporal attention through KL divergence. It is
    independent of clinical outcome grades and does not smooth their values.
    Existing numerical behavior (including the 1e-6 denominator) is retained.
    """
    base = torch.linspace(
        0.1,
        10.0,
        steps=time_steps,
        device=device
    )

    base = base.unsqueeze(0).repeat(batch_size, 1)

    valid_mask = 1.0 - src_key_padding_mask.float()
    masked_base = base * valid_mask

    masked_sum = masked_base.sum(dim=1, keepdim=True)

    attention_label = masked_base / (masked_sum + 1e-6)

    return attention_label
