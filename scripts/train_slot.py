# modified from https://github.com/evelinehong/slot-attention-pytorch/blob/master/train.py
import os
import argparse
from matplotlib import pyplot as plt
from slot_diffusion_policy.dataset.rlbench_slot_dataset import RlbenchSlotDataset
from slot_diffusion_policy.model.slot_attention import SlotAttentionAutoEncoder
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


@hydra.main(version_base=None, config_path='../configs', config_name='train_slot')
def main(cfg):
    output_dir = HydraConfig.get().run.dir

    criterion = nn.MSELoss()
    train_dataset = hydra.utils.instantiate(cfg.train_dataset)
    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader, dataset=train_dataset)
    val_dataset = hydra.utils.instantiate(cfg.val_dataset)
    val_dataloader = hydra.utils.instantiate(cfg.val_dataloader, dataset=val_dataset)
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(cfg.train.device)
    params = [{'params': model.parameters()}]
    optim = hydra.utils.instantiate(cfg.optim, params)
    
    wandb.login()
    run = wandb.init(
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=os.path.basename(output_dir),
        **cfg.wandb
    )

    start = time.time()
    i = 0
    for epoch in range(cfg.train.num_epochs):
        model.train()

        total_loss = 0

        for sample in tqdm(train_dataloader):
            i += 1

            if i < cfg.train.warmup_steps:
                learning_rate = cfg.optim.lr * (i / cfg.train.warmup_steps)
            else:
                learning_rate = cfg.optim.lr

            learning_rate = learning_rate * (cfg.train.decay_rate ** (
                i / cfg.train.decay_steps))

            optim.param_groups[0]['lr'] = learning_rate
            
            image = sample['image'].to(cfg.train.device)
            recon_combined, recons, masks, slots = model(image)
            loss = criterion(recon_combined, image)
            total_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 1000 == 0:
                torch.save({
                'model_state_dict': model.state_dict(),
                }, os.path.join(output_dir, f'model-e{epoch}-s{i}.ckpt'))
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
                recons = recons.reshape(-1, *recons.shape[-3:]).permute(0,3,1,2)
                recons = torchvision.utils.make_grid(recons, masks.shape[1])
                all_images = torch.cat([image, recon_combined, recons], dim=2)
                
                wandb.log({
                    'train_loss': total_loss,
                    'val_loss': val_loss,
                    'learning_rate': learning_rate,
                    'images': [wandb.Image(all_images)]
                })
                total_loss = 0

        print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
            datetime.timedelta(seconds=time.time() - start)))

if __name__ == '__main__':
    main()