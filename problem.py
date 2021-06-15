""" Transfer learning for Optical Access
"""
import os
import joblib
import numpy as np
import rampwf as rw
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).parent / 'external_imports'))

import utils

# Dataset choice
SOURCEDATA = 'city_A' # A priori
TARGETDATA = 'city_B' # Refine on

source_data_path = os.path.join('data', SOURCEDATA)
target_data_path = os.path.join('data', TARGETDATA)

# For private dataset, comment for public
# source_data_path = os.path.join('data', 'private', SOURCEDATA)
# target_data_path = os.path.join('data', 'private', TARGETDATA)

problem_title = 'Optical access network failure prediction'

_prediction_label_names = [0, 1]
MulticlassPredictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

class Predictions(MulticlassPredictions):
    def __init__(self, y_pred=None, y_true=None, n_samples=None,
                 fold_is=None):
        if fold_is is None:
            fold_is = utils.cv.CVFold._none_slice()
        if y_pred is not None:
            try:
                MulticlassPredictions.__init__(
                    self, y_pred=y_pred, fold_is=fold_is.target)
            except AttributeError:  # when called from combine
                MulticlassPredictions.__init__(self, y_pred=y_pred)
        elif y_true is not None:
            MulticlassPredictions.__init__(
                self, y_true=y_true.target, fold_is=fold_is.target)
        elif n_samples is not None:
            MulticlassPredictions.__init__(self, n_samples=n_samples)

    def set_valid_in_train(self, predictions, test_is):
        """Set a cross-validation slice."""
        try:
            MulticlassPredictions.set_valid_in_train(
                self, predictions, test_is.target)
        except:  # test
            MulticlassPredictions.set_valid_in_train(
                self, predictions, test_is)


score_types = [
    utils.score_types.AveragePrecision(name='ap', precision=3),
    utils.score_types.RecallAtK(name='rec-5', k=0.05, precision=3),
    utils.score_types.RecallAtK(name='rec-10', k=0.1, precision=3),
    utils.score_types.RecallAtK(name='rec-20', k=0.2, precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
    rw.score_types.ROCAUC(name='auc', precision=3),
]

workflow = utils.workflow.FeatureExtractorClassifier()

def _fetch_data(ramp_data_path):
    # If we have a pickle with the proprocessed data, load it
    ramp_data = joblib.load(ramp_data_path)
    data = ramp_data['data']
    labels = ramp_data['labels']
    return data, labels

def _read_data(x, y):
    """ Read and process the data from the raw pickle.

    Output:
        (x labeled, x unlabaled, x background, y labeled
    """
    # Take the background out
    idx = y == -1
    x_bkg = x[idx]
    x = x[~idx]
    y = y[~idx]

    # Split labeled / unlabeled if any
    idx = np.isnan(y)
    if np.all(~idx):
        return x, None, x_bkg, y
    else:
        return x[~idx], x[idx], x_bkg, y[~idx].astype(np.int)


def get_train_data(path='.'):
    print('Train data')
    # Source
    source_path = os.path.join(path, source_data_path, 'ramp_train.pickle')
    X_source, y_source = _fetch_data(source_path)

    # Target
    target_path = os.path.join(path, target_data_path, 'ramp_target.pickle')
    X_target, y_target = _fetch_data(target_path)

    source, _, source_bkg, y_source = _read_data(X_source, y_source)
    target, target_unlabeled, target_bkg, y_target = _read_data(
        X_target, y_target)

    X = utils.dataset.OpticalDataset(
        source, source_bkg, target, target_unlabeled, target_bkg)
    y = utils.dataset.OpticalLabels(y_source, y_target)

    print(X, y)
    return X, y

def get_test_data(path='.'):
    print('Test data')
    test_path = os.path.join(path, target_data_path, 'ramp_test.pickle')
    X_test, y_test = _fetch_data(test_path)

    test, _, test_bkg, y_test = _read_data(X_test, y_test)

    X = utils.dataset.OpticalDataset(None, None, test, None, test_bkg)
    y = utils.dataset.OpticalLabels(None, y_test)

    X = utils.dataset.OpticalDataset(None, None, test, None, test_bkg)
    y = utils.dataset.OpticalLabels(None, y_test)

    print(X, y)
    return X, y


def get_cv(X, y):
    cv = utils.cv.TLShuffleSplit(
        n_splits=10, test_size=0.2, random_state=42,
        train_size_labeled_target=20)
    return cv.split(X, y)
