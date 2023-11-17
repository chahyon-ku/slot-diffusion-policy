# https://github.com/google-research/google-research/blob/master/invariant_slot_attention/modules/invariant_attention.py
# SlotAttentionTranslScaleEquiv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class InvariantSlotAttention(nn.Module):
    def __init__(
            self,
            d_input=64,
            d_slot=64,
            d_hid = 128,
        ):
        super().__init__()
        self.scale = d_slot ** -0.5

        self.q_proj = nn.Linear(d_slot, d_slot, bias=False)
        self.k_proj = nn.Linear(d_input, d_slot, bias=False)
        self.v_proj = nn.Linear(d_input, d_slot, bias=False)
        self.grid_proj = nn.Linear(2, d_slot, bias=True)
        self.pos_mlp = nn.Sequential(
            nn.LayerNorm(d_slot),
            nn.Linear(d_slot, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_slot),
        )
        self.gru = nn.GRUCell(d_slot, d_slot)
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(d_slot),
            nn.Linear(d_slot, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_slot),
        )
        self.slot_norm = nn.LayerNorm(d_slot)

    def forward(self, inputs: torch.Tensor, slots: torch.Tensor, n_iters):
        # inputs: (B, N, d_input + 2)
        # slots: (n_slots, d_slot + 4)
        inputs, grids = inputs[..., :-2], inputs[..., -2:]
        slots, positions, scales = slots[..., :-4], slots[..., -4:-2], slots[..., -2:]
        B, N, d_input = inputs.shape
        n_slots, d_slot = slots.shape
        slots = slots.expand(B, -1, -1)
        positions = torch.clip(positions.expand(B, -1, -1), -1, 1)
        scales = torch.clip(scales.expand(B, -1, -1), 0.001, 2)
        # inputs: (B, N, d_input)
        # grids: (B, N, 2)
        # slots: (B, n_slots, d_slot)
        # positions: (B, n_slots, 2)
        # scales: (B, n_slots, 2)

        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        # k: (B, N, d_slot)
        # v: (B, N, d_slot)
    
        for i_iter in range(n_iters + 1):
            rel_grids = (grids[:, :, None] - positions[:, None]) / scales[:, None]
            rel_grids_proj = self.grid_proj(rel_grids)
            # rel_grids: (B, N, n_slots, 2)
            # rel_grids_proj: (B, N, n_slots, d_slot)
            k_pos = self.pos_mlp(k[:, :, None] + rel_grids_proj)
            v_pos = self.pos_mlp(v[:, :, None] + rel_grids_proj)
            # k_pos: (B, N, n_slots, d_slot)
            # v_pos: (B, N, n_slots, d_slot)

            slots_n = self.slot_norm(slots)
            q = self.q_proj(F.layer_norm(slots_n, slots_n.shape[1:]))
            # q: (B, n_slots, d_slot)

            dots = torch.einsum('bnd,bnsd->bns', q, k_pos) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            # attn: (B, N, n_slots)

            updates = torch.einsum('bns,bnd->bsd', attn, v_pos)
            positions = torch.einsum("bns,bnp->bsp", attn, grids)
            spread = (grids[:, :, None] - positions[:, None]) ** 2
            scales = torch.sqrt(torch.einsum("bns,bnp->bsp", attn, spread))
            scales = torch.clip(scales, 0.001, 2)
            # updates: (B, n_slots, d_slot)
            # positions: (B, n_slots, 2)
            # scales: (B, n_slots, 2)

            if i_iter < n_iters:
                slots = self.gru(
                    updates.reshape(-1, d_slot),
                    slots.reshape(-1, d_slot)
                ).reshape(B, n_slots, d_slot)
                slots = self.slot_mlp(slots)
                # slots: (B, n_slots, d_slot)
        
        output = torch.cat([slots, positions, scales], dim=-1)
        # output: (B, n_slots, d_slot + 4)
        return output

def build_grid(resolution):
    ranges = [np.linspace(-1., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(grid)

"""Adds soft positional embedding with learnable projection."""
class PositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.pos_proj = nn.Linear(2, hidden_size, bias=True)
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )
        self.grid = nn.Parameter(build_grid(resolution), requires_grad=False)

    def forward(self, inputs):
        pos = self.pos_proj(self.grid)
        output = self.output_mlp(inputs + pos)
        return output

"""Adds soft positional embedding with learnable projection."""
class RelativePositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.pos_proj = nn.Linear(2, hidden_size, bias=True)
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )
        self.grid = nn.Parameter(build_grid(resolution), requires_grad=False)

    def forward(self, inputs, positions, scales):
        pos = self.pos_proj((self.grid - positions) / scales)
        output = self.output_mlp(inputs + pos)
        return output

class Encoder(nn.Module):
    def __init__(self, resolution, d_hid, n_slots):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, d_hid, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(d_hid, d_hid, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(d_hid, d_hid, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(d_hid, d_hid, 5, stride=1, padding=2),
        )
        self.encoder_pos = PositionEmbed(d_hid, (16, 16))

        self.slots = nn.Parameter(torch.randn(n_slots, 64 + 4))
        self.slot_attention = InvariantSlotAttention()

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv_encoder(x)
        # x: (B, d_hid, H/8, W/8)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        # x: (B, N, d_input)
        x = self.slot_attention.forward(x, self.slots, 3)
        # x: (B, n_slots, d_slot + 4)
        # concat grid
        return x

class Decoder(nn.Module):
    def __init__(self, d_hid, resolution):
        super().__init__()
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(d_hid, d_hid, 5, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(d_hid, d_hid, 5, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(d_hid, d_hid, 5, stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(d_hid, d_hid, 5, stride=(1, 1), padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(d_hid, d_hid, 5, stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Linear(d_hid, 4),
        )
        self.decoder_initial_size = (16, 16)
        self.decoder_pos = RelativePositionEmbed(d_hid, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, slots, positions, scales):
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        B, n_slots, d_slot = slots.shape
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))

        x = self.decoder_pos(slots, positions, scales)
        x = x.permute(0,3,1,2)
        x = self.decoder_cnn(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(B, -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].
        
        return recon_combined, recons, masks, slots

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim, random_slots):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder = Encoder(self.resolution, self.hid_dim, self.num_slots)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder(image)  # CNN Backbone.

        # Slot Attention module.
        slots = self.slot_attention(x)
        slots, positions, scales = slots[..., :-4], slots[..., -4:-2], slots[..., -2:]
        # slots: (B, n_slots, d_slot)
        # positions: (B, n_slots, 2)
        # scales: (B, n_slots, 2)
        
        recon_combined, recons, masks, slots = self.decoder_cnn(slots, positions, scales)

        return recon_combined, recons, masks, slots