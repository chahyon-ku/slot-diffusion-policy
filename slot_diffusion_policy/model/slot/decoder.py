import torch.nn as nn
from slot_diffusion_policy.model.slot.util import ConcatGrid, ConcatRelGrid, ProjectAdd, Permute, SpatialBroadcast


def decoder_sa():
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

def decoder():
    return nn.Sequential(
        SpatialBroadcast((16, 16)), # B x C x H x W
        ConcatGrid((16, 16), (-1, 1)),
        Permute(0, 2, 3, 1), # B x H x W x C+2
        ProjectAdd(2, 64), # B x H x W x C
        Permute(0, 3, 1, 2), # B x C x H x W
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 1, 2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 1, 2),
        nn.ReLU(),
        nn.Conv2d(64, 4, 3, 1, 1),
        nn.Unflatten(0, (-1, 5))
    )

def decoder_rel():
    return nn.Sequential(
        SpatialBroadcast((16, 16)),
        ConcatRelGrid((16, 16), (-1, 1)),
        Permute(0, 2, 3, 1), # B x H x W x C+2
        ProjectAdd(2, 64), # B x H x W x C
        Permute(0, 3, 1, 2), # B x C x H x W
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 1, 2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 5, 1, 2),
        nn.ReLU(),
        nn.Conv2d(64, 4, 3, 1, 1),
        nn.Unflatten(0, (-1, 5))
    )