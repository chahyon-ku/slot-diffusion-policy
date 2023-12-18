# Run from the parent slot-diffusion-policy directory
# python scripts/policy/train_policy.py \
CUDA_VISIBLE_DEVICES=6 python slot_diffusion_policy/lib/diffusion_policy/train.py \
    --config-path=../../../configs \
    --config-name=train_slot_policy_real.yaml