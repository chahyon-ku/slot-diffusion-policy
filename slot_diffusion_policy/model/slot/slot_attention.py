from slot_diffusion_policy.model.slot.util import ProjectAdd
import torch
import torch.nn as nn
import torch.nn.functional as F


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
