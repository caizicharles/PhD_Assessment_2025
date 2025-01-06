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
Before running the following commands, modify these 2 arguments in "configs/data_preparation.yaml":<br>
raw_data_path:  {PATH_TO_REPO}/PhD_Assessment_2025/data/raw<br>
processed_data_path:  {PATH_TO_REPO}/PhD_Assessment_2025/data/processed<br>

```bash
python -m data_preparation.data_preparation -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/data_preparation.yaml
```

## Run Models
HIP:<br>
```bash
python main.py -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/base_mortality.yaml
python main.py -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/base_los.yaml
```

Baselines:<br>
Choose baseline name from [gru, transformer, retain, stagenet].
```bash
python main.py -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/{BASELINE_NAME}_mortality.yaml
python main.py -c {PATH_TO_REPO}/PhD_Assessment_2025/configs/BASELINE_NAME}_los.yaml
```

## Hyperparameter Search
HIP:<br>
```bash
python hyp_search_base.py
```

Baselines:<br>
```bash
python hyp_search_{BASELINE_NAME}.py
```
