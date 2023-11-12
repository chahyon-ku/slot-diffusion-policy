
for path in "https://diffusion-policy.cs.columbia.edu/data/experiments/"
do
    for mode in "image" "low_dim"
    do
        for task in "can_mh" "can_ph" "lift_mh" "lift_ph" "pusht" "square_mh" "square_ph" "tool_hang_ph" "transport_mh" "transport_ph"
        do
            for model in "diffusion_policy_cnn" "diffusion_policy_transformer" "ibc_dfo" "lstm_gmm"
            do
                wget $path/$mode/$task/$model/config.yaml -P data/experiments/$mode/$task/$model/
            done
        done
    done
done