import os

config_files = ['base_mortality', 'base_los']
k_vals = [2, 4, 8]
layer_vals = [1, 2]
softmax_temp = [0.5, 1.0, 2.0]
static_sizes = [[32, 32], [64, 64]]
predictor_sizes = [[32, 32]]
act_fn = ['relu', 'tanh']
lr_vals = [0.0001, 0.0005, 0.001]

for file in config_files:
    for k in k_vals:
        for layer in layer_vals:
            for temp in softmax_temp:
                for ssizes in static_sizes:
                    ssizes = " ".join(map(str, ssizes))
                    for psizes in predictor_sizes:
                        psizes = " ".join(map(str, psizes))
                        for fn in act_fn:
                            for lr in lr_vals:
                                os.system(f'python main.py \
                                        -c C:/Users/ZC/Documents/GitHub/PhD_Assessment_2025/configs/{file}.yaml \
                                            --seed 0 \
                                            --save_test {True} \
                                            --lr {lr} \
                                            --k_coeffs {k} \
                                            --dynamic_layers {layer} \
                                            --predictor_hidden_sizes {psizes} \
                                            --static_hidden_sizes {ssizes} \
                                            --softmax_temp {temp} \
                                            --activation {fn}'
                                        )
