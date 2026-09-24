"""Optional learning-rate schedules added after the original public Snap baseline.

The default ``none`` preserves the existing constant-learning-rate behavior.
Enabling a new schedule changes future training, not existing checkpoints.
"""

import torch


def build_snap_scheduler(
    optimizer,
    scheduler_type="none",
    warmup_epochs=5,
    start_factor=0.1,
    T_0=15,
    T_mult=2,
    eta_min=1e-7,
):
    """Build an epoch-based linear warm-up followed by cosine warm restarts.

    Call ``step()`` once after each training epoch. The cosine phase begins
    after ``warmup_epochs`` completed epochs and advances every epoch,
    independently of validation performance.
    """
    if scheduler_type == "none":
        return None
    if scheduler_type != "warmup_cosine":
        raise ValueError(f"Unknown Snap scheduler: {scheduler_type}")
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
