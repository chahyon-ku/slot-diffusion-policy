# Run from the parent slot-diffusion-policy directory
# python scripts/policy/train_policy.py \
python slot_diffusion_policy/lib/diffusion_policy/train.py \
    --config-path=../../../configs \
    --config-name=train_diffusion_unet_image_workspace.yaml \
    task=rlbench_closejar