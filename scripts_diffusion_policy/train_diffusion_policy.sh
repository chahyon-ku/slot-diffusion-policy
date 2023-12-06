# Run from the parent slot-diffusion-policy directory
python scripts_diffusion_policy/train_diffusion_policy.py \
    --config-dir=. \
    --config-name=train_diffusion_unet_image_workspace.yaml \
    training.device=cuda:0 \
    hydra.run.dir='outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'