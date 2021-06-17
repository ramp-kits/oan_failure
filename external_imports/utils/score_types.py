import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, f1_score)
from rampwf.score_types import ClassifierBaseScoreType, BaseScoreType


class F1_score(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='f1_score', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        return f1_score(y_true_label_index, y_pred_label_index)


class RecallAtK(BaseScoreType):
    """ Recall@k
    """
    is_lower_the_better = False
    minimum = 0
    maximum = 1

    def __init__(self, k, name='recall@k', precision=2):
        """
        k: proportion of highest probability labels to consider
        """
        self.name = name
        self.precision = precision
        self.k = k

    def __call__(self, y_true_proba, y_proba):
        # Get top-k
        n_samples = int(self.k * y_proba.shape[0])
        topk_idx = np.argsort(y_proba)[-n_samples:]
        return y_true_proba[topk_idx].sum() / y_true_proba.sum()

    def score_function(self, ground_truths, predictions):
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)


class AveragePrecision(BaseScoreType):
    """ Average Precision score
    """
    is_lower_the_better = False
    minimum = 0
    maximum = 1

    def __init__(self, name='AP', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        return average_precision_score(y_true_proba, y_proba)

    def score_function(self, ground_truths, predictions):
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)


class PrecisionAtRecall(BaseScoreType):
    """ Average Precision score
    """
    is_lower_the_better = False
    minimum = 0
    maximum = 1

    def __init__(self, name='Precision@recall', recall=0.2, precision=2):
        self.name = name
        self.precision = precision
        self.rec = recall

    def score_function(self, ground_truths, predictions):
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true_proba, y_proba):
        precision, recall, thresholds = precision_recall_curve(
            y_true_proba, y_proba)

        # Get precision when fixed recall
        idx = np.where(recall < self.rec)[0]
        return precision[idx[0]]
