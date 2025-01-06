import os

config_files = ['transformer_mortality', 'transformer_los']
lr_vals = [0.0001, 0.0005, 0.001]
encoder_depths = [1, 2, 3]

for file in config_files:
    for depth in encoder_depths:
        for lr in lr_vals:
            os.system(f'python main.py \
                    -c C:/Users/ZC/Documents/GitHub/PhD_Assessment_2025/configs/{file}.yaml \
                        --seed 0 --save_test {True} --lr {lr} --encoder_depth {depth}')
