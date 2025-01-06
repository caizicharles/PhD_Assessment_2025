import numpy as np
from torch.utils import data


class MIMICBaseDataset(data.Dataset):

    def __init__(self, data: dict, task: str):

        self.task = task
        self.dataset = self.extract_data(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def extract_data(self, data: dict):

        dataset = []
        if self.task == 'mortality_prediction':
            label_key = 'mortality_label'
        elif self.task == 'los_prediction':
            label_key = 'los_label'

        for patient_data in data.values():
            label = patient_data[label_key]
            subject_id = patient_data['subject_id']
            x_static = patient_data['demographic']
            x_interv = patient_data['interventions']
            x_coeffs = [data['fourier_coeffs'] for data in patient_data['vitals_labs_mean'].values()]
            x_coeffs = np.array(x_coeffs)
            
            for idx in range(x_coeffs.shape[1]):
                sample_data = {}
                sample_x_coeffs = x_coeffs[:, idx]
                sample_x_interv = x_interv[:, idx]

                sample_data['static'] = x_static.astype(np.float32)
                sample_data['dynamic'] = sample_x_interv.astype(np.float32)
                sample_data['fourier_coeffs'] = sample_x_coeffs.astype(np.complex64)
                sample_data['label'] = label.astype(np.float32)
                sample_data['subject_id'] = subject_id
                sample_data['order_id'] = idx

                dataset.append(sample_data)

        return dataset
