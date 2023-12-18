# Run from the parent slot-diffusion-policy directory
# python scripts/policy/train_policy.py \
unset QT_QPA_PLATFORM_PLUGIN_PATH
CUDA_VISIBLE_DEVICES=7 python slot_diffusion_policy/lib/diffusion_policy/train.py \
    --config-path=../../../configs \
    --config-name=train_diffusion_unet_real_image_workspace.yaml \
    task=real_pusht_image