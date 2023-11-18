import torch
import torch.nn as nn
import torch.nn.functional as F


def encoder():
    return nn.Sequential(
        nn.Conv2d(3, 64, 5, 1, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5, 1, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5, 1, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5, 1, 2), # B x C x H x  W
        ConcatGrid((128, 128), (-1, 1)), # B x C+2 x H x W
        Permute(0, 2, 3, 1), # B x H x W x C+2
        ProjectAdd(2, 64), # B x H x W x C
        nn.Flatten(1, 2), # B x H*W x C
        nn.LayerNorm(64),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
    )

def encoder_small():
    return nn.Sequential(
        nn.Conv2d(3, 64, 5, 2, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5, 2, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5, 2, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 5, 1, 2), # B x C x H x  W
        ConcatGrid((16, 16), (-1, 1)), # B x C+2 x H x W
        Permute(0, 2, 3, 1), # B x H x W x C+2
        ProjectAdd(2, 64), # B x H x W x C
        nn.Flatten(1, 2), # B x H*W x C
        nn.LayerNorm(64),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
    )

def decoder():
    return nn.Sequential(
        SpatialBroadcast((8, 8)),
        ConcatGrid((8, 8), (-1, 1)),
        Permute(0, 2, 3, 1), # B x H x W x C+2
        ProjectAdd(2, 64), # B x H x W x C
        Permute(0, 3, 1, 2), # B x C x H x W
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 1, 2),
        nn.ReLU(),
        nn.Conv2d(64, 4, 3, 1, 1),
        nn.Unflatten(0, (-1, 5))
    )

class SlotAttention(nn.Module):
    def __init__(
            self,
            in_features,
            num_iterations,
            num_slots,
            slot_size,
            mlp_hidden_size,
            epsilon=1e-8
        ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        # Learnable mu and log_sigma
        self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size + 4)), gain=nn.init.calculate_gain("linear")))
        self.slots_log_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size + 4)), gain=nn.init.calculate_gain("linear")))

    def forward(self, inputs):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = torch.randn((batch_size, self.num_slots, self.slot_size + 4))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        slots, positions, scales = slots.split([self.slot_size, 2, 2], dim=-1)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

class SlotAutoencoder(nn.Module):
    def __init__(self, encoder, slot_attention, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.slot_attention = slot_attention
        self.decoder = decoder
    
    def forward(self, x):
        encoded = self.encoder(x)
        slots = self.slot_attention(encoded)
        decoded = self.decoder(slots)
        recons, masks = torch.split(decoded, [3, 1], dim=2)
        masks = torch.softmax(masks, dim=1)
        recons_combined = torch.sum(recons * masks, dim=1)
        return recons_combined, recons, masks, slots


class ConcatGrid(nn.Module):
    def __init__(self, resolution, min_max) -> None:
        super().__init__()
        self.register_buffer('grid', build_grid(resolution, min_max)) # (1, H, W, 2)

    def forward(self, x):
        grid = self.grid.expand(x.shape[0], -1, -1, -1)
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