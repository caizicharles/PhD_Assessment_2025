import os

config_files = ['gru_mortality', 'gru_los']
lr_vals = [0.0001, 0.0005, 0.001]
layer_vals = [1, 2, 3]

for file in config_files:
    for layer in layer_vals:
        for lr in lr_vals:
            os.system(f'python main.py \
                    -c C:/Users/ZC/Documents/GitHub/PhD_Assessment_2025/configs/{file}.yaml \
                        --seed 0 --save_test {True} --lr {lr} --layer_num {layer}')
