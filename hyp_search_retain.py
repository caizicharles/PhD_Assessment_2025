import os

config_files = ['retain_mortality', 'retain_los']
lr_vals = [0.0001, 0.0005, 0.001]

for file in config_files:
    for lr in lr_vals:
        os.system(f'python main.py \
                -c C:/Users/ZC/Documents/GitHub/PhD_Assessment_2025/configs/{file}.yaml \
                    --seed 0 --save_test {True} --lr {lr}')
