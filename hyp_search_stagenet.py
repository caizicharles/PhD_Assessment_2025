import os

config_files = ['stagenet_mortality', 'stagenet_los']
conv_size = [1, 2, 4]
levels = [1, 2]
lr_vals = [0.0001, 0.0005, 0.001]

for file in config_files:
    for size in conv_size:
        for level in levels:
            for lr in lr_vals:
                os.system(f'python main.py \
                        -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/{file}.yaml \
                            --seed 0 --save_test {True} --lr {lr} --conv_size {size} --levels {level}')
