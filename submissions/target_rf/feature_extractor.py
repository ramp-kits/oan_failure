import numpy as np

class FeatureExtractor:

    def __init__(self):
        pass

    def transform(self, X):
        # Deal with NaNs inplace
        np.nan_to_num(X, copy=False)
        # We flatten the input, originally 3D (sample, time, dim) to 
        # 2D (sample, time * dim)
        X = X.reshape(X.shape[0], -1)
        return X
