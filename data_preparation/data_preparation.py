import os.path as osp
import logging
from copy import deepcopy
import random
from tqdm import tqdm
import numpy as np
from scipy import interpolate
import pandas as pd
import torch

from utils.misc import init_logger
from utils.args import get_args
from utils.utils import *

logger = logging.getLogger()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_data(dataset_max_hours: int,
              raw_data_path: str,
              processed_data_path: str,
              dataset_name: str,
              save_data: bool = False):

    if dataset_name == 'mimiciii':
        raw_data_path = osp.join(raw_data_path, 'all_hourly_data.h5')
        patient_demographics = pd.read_hdf(raw_data_path, 'patients')
        vitals_labs_mean = pd.read_hdf(raw_data_path, 'vitals_labs_mean')
        interventions = pd.read_hdf(raw_data_path, 'interventions')

        PATIENT_DEMO_KEYS = [
            "age",
            "ethnicity",
            "gender",
            "first_careunit",
            "mort_icu",
            "los_icu"
        ]

        VITALS_LABS_KEYS = [
            "diastolic blood pressure",
            "mean blood pressure",
            "heart rate"
        ]

        INTERVENTIONS_KEYS = [
            'vent',
            'vaso',
            'adenosine',
            'dobutamine',
            'dopamine',
            'epinephrine',
            'isuprel',
            'milrinone',
            'norepinephrine',
            'phenylephrine',
            'vasopressin',
            'colloid_bolus',
            'crystalloid_bolus',
            'nivdurations'
        ]

        subject_ids = patient_demographics.index.get_level_values("subject_id").to_numpy()
        icu_stay_ids = patient_demographics.index.get_level_values("icustay_id").to_numpy()

        patient_demographics = patient_demographics.reset_index()
        patient_demographics = patient_demographics[PATIENT_DEMO_KEYS]
        patient_demographics = patient_demographics.to_numpy(dtype=str)
        
        vitals_labs_mean = vitals_labs_mean[VITALS_LABS_KEYS]
        vitals_labs_mean = vitals_labs_mean.unstack(level="hours_in").to_numpy(dtype=float)
        vitals_labs_mean = vitals_labs_mean.reshape(-1, len(VITALS_LABS_KEYS), dataset_max_hours)

        interventions = interventions[INTERVENTIONS_KEYS]
        interventions = interventions.unstack(level="hours_in").to_numpy(dtype=float)
        interventions = interventions.reshape(-1, len(INTERVENTIONS_KEYS), dataset_max_hours).transpose(0, 2, 1)

        raw_data = {}
        for idx, subject_id in enumerate(subject_ids):
            age = float(patient_demographics[idx][0])
            ethnicity = patient_demographics[idx][1]
            gender = patient_demographics[idx][2]
            first_careunit = patient_demographics[idx][3]
            mort_icu = float(patient_demographics[idx][4])
            los_icu = float(patient_demographics[idx][5])

            diastolic_bp = vitals_labs_mean[idx][0]
            mean_bp = vitals_labs_mean[idx][1]
            hr = vitals_labs_mean[idx][2]

            raw_data[subject_id] = {'subject_id': subject_id,
                                    'icustay_id': icu_stay_ids[idx],
                                    'demographic': {'age': age,
                                                    'ethnicity': ethnicity,
                                                    'gender': gender,
                                                    'first_careunit': first_careunit,
                                                    'mort_icu': mort_icu,
                                                    'los_icu': los_icu},
                                    'vitals_labs_mean': {'diastolic_bp': diastolic_bp,
                                                        'mean_bp': mean_bp,
                                                        'hr': hr},
                                    'interventions': interventions[idx]}

    if save_data:
        save_with_pickle(raw_data, processed_data_path, f'{dataset_name}_raw_data.pickle')

    return raw_data


def filter_data(raw_data: dict,
                age_threshold_low: int,
                age_threshold_high: int,
                hours_threshold_low: int,
                hours_threshold_high: int,
                max_consec_nan: int,
                processed_data_path: str,
                dataset_name: str,
                save_data: bool = False):
    
    def consecutive_nan_lengths(arr):
        is_nan = np.isnan(arr)
        nan_lengths = []
        current_count = 0
        
        for val in is_nan:
            if val:
                current_count += 1
            else:
                if current_count > 0:
                    nan_lengths.append(current_count)
                    current_count = 0

        if current_count > 0:
            nan_lengths.append(current_count)
        
        return nan_lengths
    
    filtered_data = deepcopy(raw_data)

    for subject_id, data in tqdm(raw_data.items(), desc='Filtering data'):
        age = data['demographic']['age']
        subject_time_series = data['vitals_labs_mean']

        if age > age_threshold_high or age < age_threshold_low:
            del filtered_data[subject_id]
            continue

        max_intermit_nan = []
        valid_length = []
        for time_series in subject_time_series.values():
            consec_nan = consecutive_nan_lengths(time_series)
            consec_nan = np.sort(consec_nan)
            if len(consec_nan) > 1:
                max_intermit_nan.append(consec_nan[-2])
            else:
                max_intermit_nan.append(0)
            valid_length.append(len(time_series) - consec_nan[-1])
        
        if min(valid_length) > hours_threshold_high or min(valid_length) < hours_threshold_low:
            del filtered_data[subject_id]
            continue

        if max(max_intermit_nan) > max_consec_nan:
            del filtered_data[subject_id]

    if save_data:
        save_with_pickle(filtered_data, processed_data_path, f'{dataset_name}_filtered_data.pickle')

    return filtered_data


def impute_by_interpolate(filtered_data: dict,
                          dataset_max_hours: int,
                          interpolate_method: str,
                          processed_data_path: str,
                          dataset_name: str,
                          save_data: bool = False):
    
    def count_trailing_nans(arr):
        reversed_arr = arr[::-1]
        count = 0
        
        for val in reversed_arr:
            if np.isnan(val):
                count += 1
            else:
                break
        return count
    
    def interpolate_time_series(data, interpolate_method='linear'):
        nans = np.isnan(data)
        t = np.arange(len(data))
        f = interpolate.interp1d(t[~nans], data[~nans], kind=interpolate_method, fill_value='extrapolate')
        data[nans] = f(t[nans])
        return data
    
    for subject_id, data in tqdm(filtered_data.items(), desc='Imputing data'):
        subject_time_series = data['vitals_labs_mean']

        for name, time_series in subject_time_series.items():
            trailing_nan_num = count_trailing_nans(time_series)
            valid_num = len(time_series) - trailing_nan_num

            if valid_num < dataset_max_hours:
                valid_time_series = time_series[:valid_num]
            else:
                valid_time_series = time_series

            imputed_valid_time_series = interpolate_time_series(valid_time_series, interpolate_method)
            filtered_data[subject_id]['vitals_labs_mean'][name] = imputed_valid_time_series

    if save_data:
        save_with_pickle(filtered_data, processed_data_path, f'{dataset_name}_imputed_data.pickle')
    
    return filtered_data


def construct_data(imputed_data: dict,
                   window_size: int,
                   window_increment: int,
                   processed_data_path: str,
                   dataset_name: str,
                   save_data: bool = False):

    PATIENT_TEMPLATE = {
        'subject_id': None,
        'icustay_id': None,
        'demographic': None,
        'interventions': None,
        'vitals_labs_mean': {},
        'mortality_label': None
    }

    GENDER_MAP = {
        'M': 0,
        'F': 1
    }
    
    ETHNICITY_MAP = {
        'AMERICAN INDIAN/ALASKA NATIVE': 1,
        'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE': 2,
        'ASIAN': 3,
        'ASIAN - ASIAN INDIAN': 4,
        'ASIAN - CAMBODIAN': 5,
        'ASIAN - CHINESE': 6,
        'ASIAN - FILIPINO': 7,
        'ASIAN - JAPANESE': 8,
        'ASIAN - KOREAN': 9,
        'ASIAN - OTHER': 10,
        'ASIAN - THAI': 11,
        'ASIAN - VIETNAMESE': 12,
        'BLACK/AFRICAN': 13,
        'BLACK/AFRICAN AMERICAN': 14,
        'BLACK/CAPE VERDEAN': 15,
        'BLACK/HAITIAN': 16,
        'CARIBBEAN ISLAND': 17,
        'HISPANIC OR LATINO': 18,
        'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)': 19,
        'HISPANIC/LATINO - COLOMBIAN': 20,
        'HISPANIC/LATINO - CUBAN': 21,
        'HISPANIC/LATINO - DOMINICAN': 22,
        'HISPANIC/LATINO - GUATEMALAN': 23,
        'HISPANIC/LATINO - HONDURAN': 24,
        'HISPANIC/LATINO - MEXICAN': 25,
        'HISPANIC/LATINO - PUERTO RICAN': 26,
        'HISPANIC/LATINO - SALVADORAN': 27,
        'MIDDLE EASTERN': 28,
        'MULTI RACE ETHNICITY': 29,
        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 30,
        'PORTUGUESE': 31,
        'SOUTH AMERICAN': 32,
        'WHITE': 33,
        'WHITE - BRAZILIAN': 34,
        'WHITE - EASTERN EUROPEAN': 35,
        'WHITE - OTHER EUROPEAN': 36,
        'WHITE - RUSSIAN': 37,
        'OTHER': 0,
        'UNABLE TO OBTAIN': 0,
        'UNKNOWN/NOT SPECIFIED': 0,
        'PATIENT DECLINED TO ANSWER': 0
    }

    FIRST_CAREUNIT_MAP = {
        'CCU': 0,
        'CSRU': 1,
        'MICU': 2,
        'SICU': 3,
        'TSICU': 4
    }

    def get_max_time(time_series_length, window_size, window_increment):
        k = (time_series_length - window_size) // window_increment
        max_time = k * window_increment + window_size
        return max_time

    def get_fourier_freqs_and_coeffs(time_series):
        fs = 1 / 3600
        N = len(time_series)
        coeffs = np.fft.fft(time_series)
        freqs = np.fft.fftfreq(N, 1 / fs)
        
        return freqs, coeffs
    
    def discretize_los(value, max_value=10):
        
        if value >= 7:
            label = 7
        elif value >= 6 and value < 7:
            label = 6
        elif value >= 5 and value < 6:
            label = 5
        elif value >= 4 and value < 5:
            label = 4
        elif value >= 3 and value < 4:
            label = 3
        elif value >= 2 and value < 3:
            label = 2
        elif value >= 1 and value < 2:
            label = 1
        elif value > 0 and value < 1:
            label = 0
        else:
            raise ValueError('Invalid los value')
        
        return label

    max_age = 0.
    for subject_id, data in imputed_data.items():
        age = data['demographic']['age']
        if age > max_age:
            max_age = age

    constructed_data = {}

    for subject_id, data in tqdm(imputed_data.items(), desc='Constructing data'):
        this_patient = deepcopy(PATIENT_TEMPLATE)
        this_patient['subject_id'] = int(subject_id)
        this_patient['icustay_id'] = int(data['icustay_id'])

        demographic = data['demographic']
        constructed_demographic = [demographic['age'] / max_age]
        ethnicity_feats = [0] * len(ETHNICITY_MAP)
        gender_feats = [0] * len(GENDER_MAP)
        first_careunit_feats = [0] * len(FIRST_CAREUNIT_MAP)
        ethnicity_feats[ETHNICITY_MAP[demographic['ethnicity']]] = 1
        gender_feats[GENDER_MAP[demographic['gender']]] = 1
        first_careunit_feats[FIRST_CAREUNIT_MAP[demographic['first_careunit']]] = 1
        constructed_demographic.extend(ethnicity_feats)
        constructed_demographic.extend(gender_feats)
        constructed_demographic.extend(first_careunit_feats)
        this_patient['demographic'] = np.array(constructed_demographic, dtype=float)

        los_label = discretize_los(demographic['los_icu'])
        this_patient['los_label'] = np.array([los_label], dtype=float)
        this_patient['mortality_label'] = np.array([demographic['mort_icu']], dtype=float)
        
        subject_time_series = data['vitals_labs_mean']

        this_patient_max_hours = []
        for time_series in subject_time_series.values():
            this_patient_max_hours.append(get_max_time(len(time_series), window_size, window_increment))
        this_patient_max_hour = min(this_patient_max_hours)

        interventions = data['interventions']
        interventions = interventions[:this_patient_max_hour]
        assert not np.isnan(interventions).any()

        windowed_interventions = []
        for intv in interventions.T:
            windowed_intv = np.lib.stride_tricks.sliding_window_view(intv, window_size)[::window_increment]
            windowed_interventions.append(windowed_intv)
        this_patient['interventions'] = np.array(windowed_interventions)

        for name, time_series in subject_time_series.items():
            time_series = time_series[:this_patient_max_hour]
            windowed_time_series = np.lib.stride_tricks.sliding_window_view(time_series, window_size)[::window_increment]
            assert windowed_time_series.ndim == 2
        
            all_fourier_coeffs = []
            for ts in windowed_time_series:
                freqs, coeffs = get_fourier_freqs_and_coeffs(ts)
                all_fourier_coeffs.append(coeffs)

            all_fourier_coeffs = np.stack(all_fourier_coeffs)
            assert not np.isnan(all_fourier_coeffs).any()
            this_patient['vitals_labs_mean'][name] = {'windowed_data': windowed_time_series, 'fourier_coeffs': all_fourier_coeffs}

        constructed_data[subject_id] = this_patient        

    if save_data:
        save_with_pickle(constructed_data, processed_data_path, f'{dataset_name}_constructed_data.pickle')

    return constructed_data


def split_data(constructed_data: dict,
               split_ratio: tuple,
               processed_data_path: str,
               dataset_name: str,
               save_data: bool = False):
    
    patient_num = len(constructed_data)
    patients_k = np.array(list(constructed_data.keys()))

    np.random.shuffle(patients_k)

    train_num = int(split_ratio[0] * patient_num)
    val_num = int(split_ratio[1] * patient_num)

    train_k = patients_k[:train_num]
    val_k = patients_k[train_num:train_num + val_num]
    test_k = patients_k[train_num + val_num:]

    train_patients = {key: constructed_data[key] for key in train_k}
    val_patients = {key: constructed_data[key] for key in val_k}
    test_patients = {key: constructed_data[key] for key in test_k}

    if save_data:
        save_with_pickle(train_patients, processed_data_path,
                            f'{dataset_name}_train_data.pickle')
        save_with_pickle(val_patients, processed_data_path, f'{dataset_name}_val_data.pickle')
        save_with_pickle(test_patients, processed_data_path,
                            f'{dataset_name}_test_data.pickle')
    
    return {'train': train_patients, 'val': val_patients, 'test': test_patients}

def run(args):
    init_logger()
    seed_everything(args.seed)

    processed_data_path = osp.join(args.processed_data_path, args.dataset)

    raw_data = load_data(dataset_max_hours=args.dataset_max_hours,
              raw_data_path=args.raw_data_path,
              processed_data_path=processed_data_path,
              dataset_name=args.dataset,
              save_data=False)
    logger.info('Raw data loaded')
    
    filtered_data = filter_data(raw_data=raw_data,
                                age_threshold_low=args.age_threshold_low,
                                age_threshold_high=args.age_threshold_high,
                                hours_threshold_low=args.hours_threshold_low,
                                hours_threshold_high=args.hours_threshold_high,
                                max_consec_nan=args.max_consec_nan,
                                processed_data_path=processed_data_path,
                                dataset_name=args.dataset,
                                save_data=False)
    logger.info(f'raw data size {len(raw_data)} -> filtered data size {len(filtered_data)}')
    logger.info('Filtered data')

    imputed_data = impute_by_interpolate(filtered_data=filtered_data,
                                         dataset_max_hours=args.dataset_max_hours,
                                         interpolate_method=args.interpolate_method,
                                         processed_data_path=processed_data_path,
                                         dataset_name=args.dataset,
                                         save_data=False)
    logger.info('Imputed data')

    constructed_data = construct_data(imputed_data=imputed_data,
                                      window_size=args.window_size,
                                      window_increment=args.window_increment,
                                      processed_data_path=processed_data_path,
                                      dataset_name=args.dataset,
                                      save_data=False)
    logger.info('Constructed data')

    split_ratio = (args.train_proportion, args.val_proportion, args.test_proportion)
    splitted_data = split_data(constructed_data=constructed_data,
                               split_ratio=split_ratio,
                               processed_data_path=processed_data_path,
                               dataset_name=args.dataset,
                               save_data=True)
    logger.info(f'# of train patients: {len(splitted_data["train"])}')
    logger.info(f'# of val patients: {len(splitted_data["val"])}')
    logger.info(f'# of test patients: {len(splitted_data["test"])}')
    logger.info('Splitted data')

if __name__ == '__main__':
    args = get_args()
    run(args=args)
