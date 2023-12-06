import torch.nn as nn
import torch.nn.functional as F
from  slot_diffusion_policy.model.slot.util import ConcatGrid, ProjectAdd, Permute
import torchvision
from functools import partial


def encoder_sa():
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
        nn.Flatten(1, 2), # B x H*W x C+2
    )

def encoder():
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
        nn.Flatten(1, 2), # B x H*W x C+2
    )

def encoder_resnet18():
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
        ConcatGrid((16, 16), (-1, 1)), # B x C+2 x H x W,
        Permute(0, 2, 3, 1), # B x H x W x C+2
        nn.Flatten(1, 2), # B x H*W x C+2
    )

def encoder_resnet34(image_size=(128, 128)):
    latent_size = ((image_size[1] // 8), (image_size[0] // 8))
    resnet34 = torchvision.models.resnet34(pretrained=False, norm_layer=partial(nn.GroupNorm, 32))
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        resnet34.bn1,
        resnet34.relu,
        # resnet34.maxpool,
        resnet34.layer1,
        resnet34.layer2,
        resnet34.layer3,
        resnet34.layer4,
        ConcatGrid(latent_size, (-1, 1)), # B x C+2 x H x W,
        Permute(0, 2, 3, 1), # B x H x W x C+2
        nn.Flatten(1, 2), # B x H*W x C+2
    )