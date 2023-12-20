import torchvision
import torch.nn as nn
import torch
from torch.nn import Flatten
from functools import partial
import torch.nn.functional as F


class SlotImageEncoder(nn.Module):
    def __init__(self, encoder, slot_attention, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.slot_attention = slot_attention
        self.decoder = decoder
    
    def forward(self, x):
        encoded = self.encoder(x)
        slots = self.slot_attention(encoded)
        decoded = self.decoder(slots)
        return decoded

def encoder_resnet18(image_size=(128, 128)):
    latent_size = ((image_size[1] // 8), (image_size[0] // 8))
    resnet18 = torchvision.models.resnet18(pretrained=False, norm_layer=partial(nn.GroupNorm, 32))
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        resnet18.bn1,
        resnet18.relu,
        # resnet34.maxpool,
        resnet18.layer1,
        resnet18.layer2,
        resnet18.layer3,
        resnet18.layer4,
        ConcatGrid(latent_size, (-1, 1)), # B x C+2 x H x W,
        Permute(0, 2, 3, 1), # B x H x W x C+2
        nn.Flatten(1, 2), # B x H*W x C+2
    )

class ConcatGrid(nn.Module):
    def __init__(self, resolution, min_max) -> None:
        super().__init__()
        self.register_buffer('grid', build_grid(resolution, min_max)) # (1, 2, H, W)

    def forward(self, x):
        grid = self.grid.expand(x.shape[0], -1, -1, -1)
        return torch.cat([x, grid], dim=1)
    
class Permute(nn.Module):
    def __init__(self, *dims) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

def build_grid(resolution, min_max):
    ranges = [torch.linspace(min_max[0], min_max[1], steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=0)
    grid = grid[None]
    return grid


class SlotAttention(nn.Module):
    def __init__(
            self,
            in_features,
            num_iterations,
            num_slots,
            slot_size,
            mlp_hidden_size,
            learnable_slots,
            epsilon=1e-8
        ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.learnable_slots = learnable_slots

        self.norm_inputs = nn.LayerNorm(self.in_features)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        # Learnable mu and log_sigma
        if self.learnable_slots:
            self.slots = nn.Parameter(torch.randn((1, self.num_slots, self.slot_size)))
        else:
            self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")))
            self.slots_log_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")))
            
        # positional encoding
        self.pos_enc = nn.Sequential(
            ProjectAdd(2, self.slot_size), # B x H*W x C
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

    def forward(self, inputs):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs, grids = inputs.split([inputs_size - 2, 2], dim=-1)
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        k_pos = self.pos_enc(torch.cat([k, grids], dim=-1))
        v_pos = self.pos_enc(torch.cat([v, grids], dim=-1))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        if self.learnable_slots:
            slots = self.slots.expand(batch_size, -1, -1)
        else:
            slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
            slots_init = slots_init.type_as(inputs)
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k_pos, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v_pos)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            # print(updates.shape)
            # print(slots_prev.shape)
            slots = self.gru(
                updates.reshape(batch_size * self.num_slots, self.slot_size),
                slots_prev.reshape(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.reshape(batch_size, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class ProjectAdd(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return x[..., :-self.in_dim] + self.proj(x[..., -self.in_dim:])


def decoder_flatten():
    # (B, N, S)
    return Flatten()