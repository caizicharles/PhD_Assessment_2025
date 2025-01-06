# HIP: Enhance Healthcare Decision Making with Interpretable Predictions

## Step 1: Download Data
Our data preprocessing pipeline is constructed based on processed data by [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract), download the all_hourly_data.h5 from their repo.

## Step 2: Setup Environment
```bash
conda env create -f environment.yaml
```

## Step 3: Run Data preparation
Before running the following commands, modify these 2 arguments in "PhD_Assessments_2025/configs/data_preparation.yaml":<br>
raw_data_path:  {PATH_TO_REPO}/PhD_Assessments_2025/data/raw<br>
processed_data_path:  {PATH_TO_REPO}/PhD_Assessments_2025/data/processed<br>

```bash
cd {PATH_TO_REPO}/PhD_Assessments_2025
python -m data_preparation.data_preparation -c {PATH_TO_REPO}/PhD_Assessments_2025/configs/data_preparation.yaml
```
