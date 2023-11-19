# modified from https://github.com/evelinehong/slot-attention-pytorch/blob/master/train.py
import os
import argparse
from matplotlib import pyplot as plt
from slot_diffusion_policy.dataset.rlbench_slot_dataset import RlbenchSlotDataset
from torch import nn
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import wandb
import torchvision
from slot_diffusion_policy.model.slot.invariant_slot_attention import InvariantSlotAttention


@hydra.main(version_base=None, config_path='../configs', config_name='train_slot')
def main(cfg):
    output_dir = HydraConfig.get().run.dir

    criterion = nn.MSELoss()
    train_dataset = hydra.utils.instantiate(cfg.train_dataset)
    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader, dataset=train_dataset)
    val_dataset = hydra.utils.instantiate(cfg.val_dataset)
    val_dataloader = hydra.utils.instantiate(cfg.val_dataloader, dataset=val_dataset)
    model = hydra.utils.instantiate(cfg.slot_model)
    model = model.to(cfg.train.device)
    params = [{'params': model.parameters()}]
    optim = hydra.utils.instantiate(cfg.optim, params)
    scheduler = hydra.utils.instantiate(cfg.scheduler, optim)
    
    wandb.login()
    run = wandb.init(
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=os.path.basename(output_dir),
        **cfg.wandb
    )

    total_loss = 0
    n_totals = 0
    train_dataloder_iter = iter(train_dataloader)
    for step in tqdm(list(range(cfg.train.train_steps))):
        try:
            batch = next(train_dataloder_iter)
        except StopIteration:
            train_dataloder_iter = iter(train_dataloader)
            batch = next(train_dataloder_iter)

        if 'transport' in cfg.slot_model._target_:
            src = batch[0]['image'].to(cfg.train.device)
            image = batch[1]['image'].to(cfg.train.device)
            recon_combined, recons, masks, slots = model(src, image)
        else:
            image = batch['image'].to(cfg.train.device)
            recon_combined, recons, masks, slots = model(image)
        loss = criterion(recon_combined, image)
        total_loss += loss.item()
        n_totals += 1

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        if step % cfg.train.f_eval == 0:
            torch.save({
            'model_state_dict': model.state_dict(),
            }, os.path.join(output_dir, f'model-s{step}.ckpt'))
            
            # display reconsturcted images and each slots
            image = image[:4]
            recon_combined = recon_combined[:4]
            recons = recons[:4]
            masks = masks[:4]
            image = torchvision.utils.make_grid(image, 1)
            recon_combined = torchvision.utils.make_grid(recon_combined, 1)
            recons = recons * masks + (1 - masks)
            recons = recons.reshape(-1, *recons.shape[-3:])
            recons = torchvision.utils.make_grid(recons, masks.shape[1])
            train_all_images = torch.cat([image, recon_combined, recons], dim=2)

            with torch.no_grad():
                val_loss = 0
                for sample in tqdm(val_dataloader):
                    image = sample['image'].to(cfg.train.device)
                    recon_combined, recons, masks, slots = model(image)
                    loss = criterion(recon_combined, image)
                    val_loss += loss.item()
            
            # display reconsturcted images and each slots
            image = image[:4]
            recon_combined = recon_combined[:4]
            recons = recons[:4]
            masks = masks[:4]
            image = torchvision.utils.make_grid(image, 1)
            recon_combined = torchvision.utils.make_grid(recon_combined, 1)
            recons = recons * masks + (1 - masks)
            recons = recons.reshape(-1, *recons.shape[-3:])
            recons = torchvision.utils.make_grid(recons, masks.shape[1])
            val_all_images = torch.cat([image, recon_combined, recons], dim=2)

            total_loss /= cfg.train.f_eval
            val_loss /= len(val_dataloader)
            
            wandb.log({
                'train_loss': total_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_lr(),
                'train_images': [wandb.Image(train_all_images)],
                'val_images': [wandb.Image(val_all_images)],
            })
            total_loss = 0
            n_totals = 0
            
            

if __name__ == '__main__':
    main()