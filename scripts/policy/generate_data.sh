for i in 128 256
do
    for j in {0..9}
    do
        python slot_diffusion_policy/lib/RLBench/tools/dataset_generator.py \
            --save_path=data/rlbench_$i/train \
            --tasks=close_jar \
            --image_size=$i,$i \
            --renderer=opengl3 \
            --processes=1 \
            --episodes_per_task=10 \
            --variations=1 \
            --all_variations=False \
            --start_episode_num=$((j * 10)) &
    done
done