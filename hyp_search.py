import os

config_files = ['base_mortality']  #, 'base_los']
lr_vals = [0.0001, 0.0005, 0.001]
fuse_dims = [32, 64]
k_vals = [2, 4, 8]

for file in config_files:
    for dim in fuse_dims:
        for k in k_vals:
            for lr in lr_vals:
                os.system(f'python /Users/caizicharles/Documents/GitHub/PhD_Assessment_2025/main.py \
                        -c /Users/caizicharles/Documents/GitHub/PhD_Assessment_2025/configs/{file}.yaml \
                            --seed 0 --save_test {True} --lr {lr} --fuse_dim {dim} --k_coeffs {k}')
