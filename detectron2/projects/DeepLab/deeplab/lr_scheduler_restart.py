import math
import torch
from detectron2.solver.lr_scheduler import LRScheduler, _get_warmup_factor_at_iter
from typing import List

class WarmupPolyLR(LRScheduler):
    """
    Poly learning rate schedule with warmup, split into two segments.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,  # Total iterations x = a + b
        # segment_iters: List[int],  # [a, b]
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        a = 0.5
        self.segment_iters = [int(a*max_iters), int((1-a)*max_iters)]  # [a, b], where sum(segment_iters) == max_iters
        # assert sum(segment_iters) == max_iters, "Sum of segment_iters must be equal to max_iters"
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Determine which segment we're in
        cumulative_iters = 0
        for idx, segment_length in enumerate(self.segment_iters):
            if self.last_epoch < cumulative_iters + segment_length:
                # We're in segment idx
                segment_max_iters = segment_length
                segment_last_epoch = self.last_epoch - cumulative_iters
                break
            cumulative_iters += segment_length
        else:
            # If last_epoch exceeds max_iters, keep the last learning rate
            segment_max_iters = self.segment_iters[-1]
            segment_last_epoch = self.segment_iters[-1]

        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, segment_last_epoch, self.warmup_iters, self.warmup_factor
        )

        if self.constant_ending > 0 and warmup_factor == 1.0:
            if (
                math.pow((1.0 - segment_last_epoch / segment_max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]

        return [
            base_lr * warmup_factor * math.pow((1.0 - segment_last_epoch / segment_max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()
