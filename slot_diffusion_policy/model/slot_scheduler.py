import torch


class SlotScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optim, warmup_steps, decay_steps, decay_rate):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        super().__init__(optim)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            learning_rate = self.base_lrs[0] * (self.last_epoch / self.warmup_steps)
        else:
            learning_rate = self.base_lrs[0] * (
                self.decay_rate ** ((self.last_epoch - self.warmup_steps) / self.decay_steps)
            )

        return [learning_rate]