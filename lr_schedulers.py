from enum import Enum
import math
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from config import TrainingConfig


class LRScheduler(Enum):
    WARMUP_CONSTANT = "warmup_constant"
    WARMUP_LINEAR = "warmup_linear"
    WARMUP_COSINE = "warmup_cosine"
    INVERSE_SQRT = "inverse_sqrt"


def get_lr_scheduler(optimizer: Optimizer, config: TrainingConfig, embed_dim: int):
    if config.lr_scheduler == LRScheduler.WARMUP_CONSTANT.value:
        return WarmupConstantLR(optimizer, config.warmup_steps, config.min_lr)
    elif config.lr_scheduler == LRScheduler.WARMUP_LINEAR.value:
        return WarmupLinearLR(optimizer, config.warmup_steps, config.epochs * config.steps_per_epoch, config.min_lr)
    elif config.lr_scheduler == LRScheduler.WARMUP_COSINE.value:
        return WarmupCosineLR(optimizer, config.warmup_steps, config.epochs * config.steps_per_epoch, config.min_lr)
    elif config.lr_scheduler == LRScheduler.INVERSE_SQRT.value:
        # Adjust the scale so that learning rate peaks at config.init_lr
        base_lr = optimizer.param_groups[0]['lr']
        scale = (config.init_lr) / base_lr * (embed_dim * config.warmup_steps) ** 0.5
        return InverseSqrtLR(optimizer, config.warmup_steps, embed_dim, config.min_lr, scale)
    else:
        raise ValueError(
            f"Unknown lr_scheduler {config.lr_scheduler!r}. "
            f"Valid options: {[e.value for e in LRScheduler]}"
        )


def _with_floor(step: int, warmup_steps: int, lrs: List[float], min_lr: float) -> List[float]:
    if min_lr is None or step < warmup_steps:
        return lrs
    return [max(min_lr, x) for x in lrs]


class WarmupCosineLR(_LRScheduler):
    """
        Linear warmup to each param group's initial lr, then cosine decay to min_lr.
        Args:
            optimizer: torch optimizer
            warmup_steps: linear ramp steps
            total_steps: total scheduler steps (must be > warmup_steps)
            min_lr: floor for LR (default 0.0)
            last_epoch: for resuming (default -1)
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        assert total_steps > 0 and warmup_steps >= 0 and total_steps > warmup_steps
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        t = max(0, self.last_epoch)  # step index
        if t < self.warmup_steps:
            warm_frac = t / max(1, self.warmup_steps)
            lrs = [base * warm_frac for base in self.base_lrs]
            return _with_floor(t, self.warmup_steps, lrs, self.min_lr)
        # cosine phase — clamp progress so lr stays at min_lr past total_steps
        progress = min(1.0, (t - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps)))
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        lrs = [self.min_lr + (base - self.min_lr) * cos_factor for base in self.base_lrs]
        return _with_floor(t, self.warmup_steps, lrs, self.min_lr)


class WarmupLinearLR(_LRScheduler):
    """
        Linear warmup to initial lr, then linear decay to min_lr at total_steps.
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        assert total_steps > 0 and warmup_steps >= 0 and total_steps > warmup_steps
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        t = max(0, self.last_epoch)
        if t < self.warmup_steps:
            warm_frac = t / max(1, self.warmup_steps)
            lrs = [base * warm_frac for base in self.base_lrs]
            return _with_floor(t, self.warmup_steps, lrs, self.min_lr)
        # linear decay
        progress = (t - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
        decay = 1.0 - progress
        lrs = [self.min_lr + (base - self.min_lr) * max(0.0, decay) for base in self.base_lrs]
        return _with_floor(t, self.warmup_steps, lrs, self.min_lr)


class WarmupConstantLR(_LRScheduler):
    """
        Linear warmup to initial lr, then keep it constant (with optional floor).
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        assert warmup_steps >= 0
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        t = max(0, self.last_epoch)
        if t < self.warmup_steps:
            warm_frac = t / max(1, self.warmup_steps)
            lrs = [base * warm_frac for base in self.base_lrs]
            return _with_floor(t, self.warmup_steps, lrs, self.min_lr)
        return _with_floor(t, self.warmup_steps, self.base_lrs, self.min_lr)


class InverseSqrtLR(_LRScheduler):
    """
    'Noam' schedule (Transformer):
        lr(step) = scale * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    We multiply each param group's initial lr (base) by the factor above.
    Args:
        optimizer
        warmup_steps: warmup in steps (>=1)
        d_model: model dim used for scaling (set to your embed dim)
        scale: extra user scaling factor (default 1.0)
        min_lr: optional floor
    Notes:
        - Start steps at 1 to avoid div-by-zero.
        - Your optimizer's initial lr acts like a 'base_lr' multiplier.
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, d_model: int, min_lr: float = 0.0, scale: float = 1.0, last_epoch: int = -1):
        assert warmup_steps >= 1 and d_model > 0
        self.warmup_steps = float(warmup_steps)
        self.d_model = float(d_model)
        self.scale = float(scale)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # step counting starts at 0 in _LRScheduler; convert to 1-based
        step = max(1.0, float(self.last_epoch))
        factor = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        factor *= self.scale
        lrs = [base * factor for base in self.base_lrs]
        return _with_floor(step, self.warmup_steps, lrs, self.min_lr)
