# https://github.com/evelinehong/slot-attention-pytorch/blob/master/eval.ipynb
from slot_diffusion_policy.dataset.rlbench_slot_dataset import RlbenchSlotDataset
import matplotlib.pyplot as plt
from slot_diffusion_policy.model.slot_transport import SlotAttentionAutoEncoder
import torch
from tqdm import tqdm


if __name__ == '__main__':
    seed = 0
    batch_size = 16
    num_slots = 7
    num_iterations = 3
    resolution = (128, 128)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    with torch.inference_mode():
        model = SlotAttentionAutoEncoder(resolution, num_slots, num_iterations, 64)
        model.load_state_dict(torch.load('./models3/model100000.ckpt')['model_state_dict'])
        model = model.to(device)

        dataset = RlbenchSlotDataset(
            data_dir='/media/rpm/Data/imitation_learning/slot-diffusion-policy/data/test',
            tasks=['reach_and_drag'],
            views=['front_rgb', 'front_depth']
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for sample in tqdm(dataloader):
            image = sample['views']['front_rgb'].to(device)
            recon_combined, recons, masks, slots = model(image)
            for i_batch in range(image.shape[0]):
                fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
                curr_image = image[i_batch]
                curr_recon_combined = recon_combined[i_batch]
                curr_recons = recons[i_batch]
                curr_masks = masks[i_batch]
                curr_image = curr_image.permute(1,2,0).cpu().numpy()
                curr_recon_combined = curr_recon_combined.permute(1,2,0)
                curr_recon_combined = curr_recon_combined.cpu().detach().numpy()
                curr_recons = curr_recons.cpu().detach().numpy()
                curr_masks = curr_masks.cpu().detach().numpy()
                ax[0].imshow(curr_image)
                ax[0].set_title('Image')
                ax[1].imshow(curr_recon_combined)
                ax[1].set_title('Recon.')
                for i in range(num_slots):
                    picture = curr_recons[i] * curr_masks[i] + (1 - curr_masks[i])
                    ax[i + 2].imshow(picture)
                    ax[i + 2].set_title('Slot %s' % str(i + 1))
                for i in range(len(ax)):
                    ax[i].grid(False)
                    ax[i].axis('off')
                plt.show()
