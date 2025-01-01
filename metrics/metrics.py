import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, cohen_kappa_score


class MetricBase():

    def __init__(self, *args, **kwargs) -> None:
        pass

    def calculate(self, probability, target):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError


class AUROC(MetricBase):

    def __init__(self, task, **kwargs):

        self.NAME = 'AUROC'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return roc_auc_score(target, probability)

        elif self.task == 'los_prediction':
            present_classes = np.unique(target).astype(int)
            probability = probability[:, present_classes]
            print(np.unique(target), target.shape, probability.shape)
            return roc_auc_score(target, probability, multi_class="ovr", average="macro")

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class AUPRC(MetricBase):

    def __init__(self, task, **kwargs):

        self.NAME = 'AUPRC'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction':
            return average_precision_score(target, probability)

        elif self.task == 'los_prediction':
            return 0

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Kappa(MetricBase):

    def __init__(self, task, **kwargs):

        self.NAME = 'Kappa'
        self.task = task

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction':
            probability = np.squeeze(probability, axis=-1)
            probability = (probability >= 0.5).astype(int)
            target = np.squeeze(target, axis=-1)
            return cohen_kappa_score(target, probability)

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return cohen_kappa_score(target, pred)

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Accuracy(MetricBase):

    def __init__(self, task, **kwargs):

        self.NAME = 'Accuracy'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction':
            probability = np.squeeze(probability, axis=-1)
            pred = (probability >= 0.5).astype(int)
            target = np.squeeze(target, axis=-1)
            return accuracy_score(target, pred)

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return accuracy_score(target, pred)

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class F1(MetricBase):

    def __init__(self, task, **kwargs):

        self.NAME = 'F1'
        self.task = task

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            probability = (probability >= 0.5).astype(int)
            return f1_score(target, probability, average="macro", zero_division=1)

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return accuracy_score(target, pred)

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


METRICS = {
    'AUROC': AUROC,
    'AUPRC': AUPRC,
    'F1': F1,
    'Accuracy': Accuracy,
    'Kappa': Kappa,
}
