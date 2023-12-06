import torch.nn as nn
from slot_diffusion_policy.model.slot.util import ConcatGrid, ConcatRelGrid, ProjectAdd, Permute, SpatialBroadcast


def decoder_sa(num_slots, image_size=(128, 128)):
    latent_size = ((image_size[1] // 16), (image_size[0] // 16))
    return nn.Sequential(
        SpatialBroadcast(latent_size),
        ConcatGrid(latent_size, (-1, 1)),
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
        nn.Unflatten(0, (-1, num_slots))
    )

def decoder(num_slots, image_size=(128, 128)):
    latent_size = ((image_size[1] // 8), (image_size[0] // 8))
    return nn.Sequential(
        SpatialBroadcast(latent_size), # B x C x H x W
        ConcatGrid(latent_size, (-1, 1)),
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
        nn.Unflatten(0, (-1, num_slots))
    )

def decoder_rel(num_slots, image_size=(128, 128)):
    latent_size = ((image_size[1] // 8), (image_size[0] // 8))
    return nn.Sequential(
        SpatialBroadcast(latent_size),
        ConcatRelGrid(latent_size, (-1, 1)),
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
        nn.Unflatten(0, (-1, num_slots))
    )