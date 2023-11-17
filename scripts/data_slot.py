from slot_diffusion_policy.dataset.rlbench_slot_dataset import RlbenchSlotDataset
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = RlbenchSlotDataset(
        data_dir='/media/rpm/Data/imitation_learning/slot-diffusion-policy/data/test',
        tasks=['stack_blocks'],
        views=['front_rgb', 'front_depth']
    )
    
    for i_sample, sample in enumerate(dataset):
        print(sample['i_eps'], sample['i_obs'], sample['variation_number'])
        print(sample)
        # display rgb and depth images
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(sample['views']['front_rgb'])
        plt.subplot(1, 2, 2)
        plt.imshow(sample['views']['front_depth'])
        plt.show()