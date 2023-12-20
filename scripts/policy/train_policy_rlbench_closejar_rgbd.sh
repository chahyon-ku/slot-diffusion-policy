# Run from the parent slot-diffusion-policy directory
# python scripts/policy/train_policy.py \
CUDA_VISIBLE_DEVICES=1 python scripts/policy/train_rlbench.py \
    --config-path=../../configs \
    --config-name=image_rgbd.yaml \
    task=rlbench_closejar_rgbd