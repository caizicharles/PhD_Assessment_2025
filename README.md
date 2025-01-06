# HIP: Enhance Healthcare Decision Making with Interpretable Predictions

## Step 1: Download Data
Our data preprocessing pipeline is constructed based on processed data by [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract), download the "all_hourly_data.h5" from their repo.

## Step 2: Setup Environment
```bash
conda env create -f environment.yaml
conda activate HIP
```

## Step 3: Create Project Structure
This step creates the required directories for data, logging, viusualization, etc.
```bash
cd {PATH_TO_REPO}/PhD_Assessment_2025
python create_project_structure.py --working_dir {PATH_TO_REPO}/PhD_Assessment_2025
```

Subsequently, move "all_hourly_data.h5" under "{PATH_TO_REPO}/PhD_Assessment_2025/data/raw/mimiciii".

## Step 4: Run Data preparation
```bash
python -m data_preparation.data_preparation -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/data_preparation.yaml --raw_data_path {PATH_TO_REPO}/PhD_Assessment_2025/data/raw --processed_data_path {PATH_TO_REPO}/PhD_Assessment_2025/data/processed
```

## Run Models
HIP:<br>
```bash
python main.py -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/base_mortality.yaml --processed_data_path {PATH_TO_REPO}/PhD_Assessment_2025/data/processed --log_data_path --processed_data_path {PATH_TO_REPO}/PhD_Assessment_2025/logs
python main.py -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/base_los.yaml --processed_data_path {PATH_TO_REPO}/PhD_Assessment_2025/data/processed --log_data_path --processed_data_path {PATH_TO_REPO}/PhD_Assessment_2025/logs
```

Baselines:<br>
Choose baseline name from [gru, transformer, retain, stagenet].
```bash
python main.py -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/{BASELINE_NAME}_mortality.yaml --processed_data_path {PATH_TO_REPO}/PhD_Assessment_2025/data/processed --log_data_path --processed_data_path {PATH_TO_REPO}/PhD_Assessment_2025/logs
python main.py -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/BASELINE_NAME}_los.yaml --processed_data_path {PATH_TO_REPO}/PhD_Assessment_2025/data/processed --log_data_path --processed_data_path {PATH_TO_REPO}/PhD_Assessment_2025/logs
```

## Hyperparameter Search
Before running the script, modify the path to main file and the path to config file in the script.
HIP:<br>
```bash
python hyp_search_base.py
```

Baselines:<br>
```bash
python hyp_search_{BASELINE_NAME}.py
```

## Visualization
```bash
python -m visualization.visualize --checkpoint {PATH_TO_REPO}/PhD_Assessment_2025/logs/checkpoints/mimiciii/mortality_prediction/HIP/{CHECKPOINT_NAME}
```
