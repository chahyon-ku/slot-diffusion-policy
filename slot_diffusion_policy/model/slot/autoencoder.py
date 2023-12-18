import torch
import torch.nn as nn
import torch.nn.functional as F

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