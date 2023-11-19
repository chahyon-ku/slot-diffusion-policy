import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatGrid(nn.Module):
    def __init__(self, resolution, min_max) -> None:
        super().__init__()
        self.register_buffer('grid', build_grid(resolution, min_max)) # (1, 2, H, W)

    def forward(self, x):
        grid = self.grid.expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, grid], dim=1)

class ConcatRelGrid(nn.Module):
    def __init__(self, resolution, min_max) -> None:
        super().__init__()
        self.register_buffer('grid', build_grid(resolution, min_max)) # (1, 2, H, W)

    def forward(self, x):
        x, positions, scales = x.split([x.shape[1] - 4, 2, 2], dim=1) # (B, C, H, W) -> (B, C-4, H, W), (B, 2, H, W), (B, 2, H, W)
        grid = (self.grid - positions) / scales
        return torch.cat([x, grid], dim=1)

def build_grid(resolution, min_max):
    ranges = [torch.linspace(min_max[0], min_max[1], steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=0)
    grid = grid[None]
    return grid

class ProjectAdd(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return x[..., :-self.in_dim] + self.proj(x[..., -self.in_dim:])

class SpatialBroadcast(nn.Module):
    def __init__(self, resolution) -> None:
        super().__init__()
        self.resolution = resolution

    def forward(self, x):
        batch_size, num_slots, slot_size = x.shape

        x = x.reshape(batch_size * num_slots, slot_size, 1, 1)
        x = x.expand(-1, -1, *self.resolution)
        return x
    
class Permute(nn.Module):
    def __init__(self, *dims) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

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