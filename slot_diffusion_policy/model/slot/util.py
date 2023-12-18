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
    

# https://github.com/openai/CLIP/blob/main/clip/model.py
class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = x.permute(1, 0, 2) # NSC -> SNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)