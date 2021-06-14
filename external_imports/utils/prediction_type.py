import numpy as np
import warnings
# from rampwf.prediction_types.base import BasePrediction


class BasePrediction(object):
    def __str__(self):
        return 'y_pred = {}'.format(self.y_pred)

    @property
    def valid_indexes(self):
        """Return valid indices (e.g., a cross-validation slice).

        When combining Predictions on different cross validation slices,
        we start with an empty y_pred. Each time a fold is added, some
        entries become valid. Invalid entries are those that are not
        predicted by any folds, i.e., those that remain NaN.
        """
        if len(self.y_pred.shape) == 1:
            return ~np.isnan(self.y_pred)
        elif len(self.y_pred.shape) == 2:
            return ~np.isnan(self.y_pred[:, 0])
        else:
            raise ValueError('y_pred.shape > 2 is not implemented')

    def set_valid_in_train(self, predictions, test_is):
        """Set a cross-validation slice."""
        if isinstance(test_is, tuple): # (X, X_extra)
            test_is = test_is[0]
        self.y_pred[test_is] = predictions.y_pred

    def set_slice(self, valid_indexes):
        """Collapsing y_pred to a cross-validation slice.

        So scores do not need to deal with masks.
        """
        self.y_pred = self.y_pred[valid_indexes]

    def check_y_pred_dimensions(self):
        if self.n_columns == 0 and len(self.y_pred.shape) != 1:
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should be 1D, '
                'instead its shape is {}'.format(self.y_pred.shape))
        if self.n_columns > 0:
            if len(self.y_pred.shape) != 2 or\
                    self.y_pred.shape[1] != self.n_columns:
                raise ValueError(
                    'Wrong y_pred dimensions: y_pred should be 2D '
                    'with {} columns, instead its shape is {}'.format(
                        self.n_columns, self.y_pred.shape))

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Combine predictions in predictions_list[index_list].

        The default implemented here is by taking the mean of their y_pred
        views. It can be overridden in derived classes.

        E.g. for regression it is the actual
        predictions, and for classification it is the probability array (which
        should be calibrated if we want the best performance). Called both for
        combining one submission on cv folds (a single model that is trained on
        different folds) and several models on a single fold (blending).

        Parameters
        ----------
        predictions_list : list of instances of Base
            Each element of the list is an instance of Base with the
            same length and type.
        index_list : None | list of integers
            The subset of predictions to be combined. If None, the full set is
            combined.

        Returns
        -------
        combined_predictions : instance of cls
            A predictions instance containing the combined predictions.
        """
        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))
        y_comb_list = np.array(
            [predictions_list[i].y_pred for i in index_list])
        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y_comb = np.nanmean(y_comb_list, axis=0)
        combined_predictions = cls(y_pred=y_comb)
        return combined_predictions


def _multiclass_init(self, y_pred=None, y_true=None, n_samples=None,
                     fold_is=None):
    """Initialize a multiclass/multilabel prediction type.

    The input is either y_pred, or y_true, or n_samples.

    Parameters
    ----------
    y_pred : a 2D numpy array
        representing the predictions (probas) returned by
        problem.workflow.test_submission
    y_true : list of objects or list of list of objects
        representing the ground truth returned by problem.get_train_data
        and problem.get_test_data; list (multiclass - single label)
        or list of lists (multilabel)
    n_samples : int
        to initialize an empty container, for the combined predictions
    fold_is : a list of integers
        either the training indices, validation indices, or None when we
        use the (full) test data.
    """
    if y_pred is not None:
        if fold_is is not None:
            y_pred = y_pred[fold_is[0]] # Fold is a tuple (X, X_extra)
        self.y_pred = np.array(y_pred)
    elif y_true is not None:
        if fold_is is not None:
            y_true = y_true[fold_is[0]]
        self._init_from_pred_labels(y_true)
    elif n_samples is not None:
        self.y_pred = np.empty((n_samples, self.n_columns), dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')
    self.check_y_pred_dimensions()


def _init_from_pred_labels(self, y_pred_labels):
    """Initalize y_pred to uniform for (positive) labels in y_pred_labels.

    Initialize multiclass Predictions from ground truth. y_pred_labels
    can be a single (positive) label in which case the corresponding
    column gets probability of 1.0. In the case of multilabel (k > 1
    positive labels), the columns corresponing the positive labels
    get probabilities 1/k.

    Parameters
    ----------
    y_pred_labels : list of objects or list of list of objects
        (of the same type)
    """
    type_of_label = np.float#type(self.label_names[0])
    self.y_pred = np.zeros(
        (len(y_pred_labels), len(self.label_names)), dtype=np.float64)
    for ps_i, label_list in zip(self.y_pred, y_pred_labels):
        # converting single labels to list of labels, assumed below
        if type(label_list) != np.ndarray and type(label_list) != list:
            label_list = [label_list]
        label_list = list(map(type_of_label, label_list))
        for label in label_list:
            if np.isnan(label): continue # Throw out NaN labels
            ps_i[self.label_names.index(label)] = 1.0 / len(label_list)


@property
def _y_pred_label_index(self):
    """Multi-class y_pred is the index of the predicted label."""
    return np.argmax(self.y_pred, axis=1)


@property
def _y_pred_label(self):
    return self.label_names[self.y_pred_label_index]


@classmethod
def _combine(cls, predictions_list, index_list=None):
    if index_list is None:  # we combine the full list
        index_list = range(len(predictions_list))
    y_comb_list = np.array(
        [predictions_list[i].y_pred for i in index_list])
    # clipping probas into [0, 1], also taking care of the case of all zeros
    y_comb_list = np.clip(y_comb_list, 10 ** -15, 1 - 10 ** -15)
    # normalizing probabilities
    y_comb_list = y_comb_list / np.sum(y_comb_list, axis=2, keepdims=True)
    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_comb = np.nanmean(y_comb_list, axis=0)
    combined_predictions = cls(y_pred=y_comb)
    return combined_predictions


def make_multiclass(label_names=[]):
    Predictions = type(
        'Predictions',
        (BasePrediction,),
        {'label_names': label_names,
         'n_columns': len(label_names),
         # Multiclass ground truth is a 1D vector of labels
         'n_columns_true': 0,
         '__init__': _multiclass_init,
         '_init_from_pred_labels': _init_from_pred_labels,
         'y_pred_label_index': _y_pred_label_index,
         'y_pred_label': _y_pred_label,
         'combine': _combine,
         })
    return Predictions
