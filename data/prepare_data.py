import os
import numpy as np
import joblib
from tqdm import tqdm

NDAYS = 7


def reshape_dataset_ts(x, y, remove_nan_labels=True, n_days_past=None):
    """ Reshape dataset of shape [rows, timestamps, features] into time serie
    framework.  Aggregates the time in order to give a dataset of shape
    [more_rows, timestamps, features]
    Made for Time serie purposes.
    The output will be in the shape [sequence, dataset, features].

    Args:
    -----
    x: array (float), of shape [samples, timestamps, features]
    y: array (ternary: {-1, 0, 1}), of shape [samples, timestamps]
    n_days_past: integer, number of days in the past to consider

    Output:
    -------
    array of shape [time, users, features] aggregated from the time serie `x`
    """
    n_samples, n_timestamps, n_features = x.shape
    # Multiply 24h by 4 as the data is collected every 15mn
    n_cols = 24 * 4 * n_days_past
    dataset = None
    labels = []
    for i in tqdm(range(n_timestamps//(24*4) - n_days_past)):
        day = 24*4*i
        # day = 24*i
        z = x[:, day:day+n_cols, :]
        lab = y[:, i]
        # Remove the missing data that has been introduced in the preprocessing
        idx = ~np.all(np.isnan(z), axis=(1, 2))
        lab = lab[idx]
        z = z[idx]  # Filter the corresponding users
        if remove_nan_labels:
            idx = ~np.isnan(lab)  # Remove windows with no label
            lab = lab[idx]
            z = z[idx]  # Filter the corresponding users
        z = z.astype(np.float32)
        labels.extend(lab)
        if dataset is None:
            dataset = z
        else:
            dataset = np.concatenate((dataset, z), axis=0)
    return dataset, np.asarray(labels)


def prepare_data(train_path=None, labels_path=None, ramp_data_path=None,
                 remove_nan_labels=True):
    data = np.load(train_path)
    labels = np.load(labels_path, allow_pickle=True)

    data, labels = reshape_dataset_ts(data,
                                      labels.astype(np.float32),
                                      remove_nan_labels,
                                      n_days_past=NDAYS)
    ramp_data = {'data': data, 'labels': labels}
    with open(ramp_data_path, 'wb') as f:
        joblib.dump(ramp_data, f)

    return data, labels


if __name__ == "__main__":
    # Dataset choice
    SOURCEDATA = 'city_A'
    TARGETDATA = 'city_B'

    source_data_path = os.path.join(SOURCEDATA)
    target_data_path = os.path.join(TARGETDATA)

    # Source data
    source_path = os.path.join(source_data_path,
                               'source.npy')
    source_labels_path = os.path.join(source_data_path,
                                      'source_labels.npy')
    ramp_source_path = os.path.join(source_data_path,
                                    'ramp_train.pickle')

    # Target data
    target_path = os.path.join(target_data_path,
                               'target.npy')
    target_labels_path = os.path.join(target_data_path,
                                      'target_labels.npy')
    ramp_target_path = os.path.join(target_data_path,
                                    'ramp_target.pickle')

    # Test data
    test_path = os.path.join(target_data_path,
                             'test.npy')
    test_labels_path = os.path.join(target_data_path,
                                    'test_labels.npy')
    ramp_test_path = os.path.join(target_data_path,
                                  'ramp_test.pickle')

    print('Prepare source...', end='')
    prepare_data(source_path, source_labels_path, ramp_source_path)
    print('Ok')

    print('Prepare target...', end='')
    prepare_data(target_path, target_labels_path,
                 ramp_target_path, remove_nan_labels=False)
    print('Ok')

    print('Prepare test...', end='')
    prepare_data(test_path, test_labels_path, ramp_test_path)
    print('Ok')
