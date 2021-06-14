import numpy as np

class OpticalDataset:

    def __init__(self, source=None, source_bkg=None, target=None,
               target_unlabeled=None, target_bkg=None):
           self.source = source
           self.source_bkg = source_bkg
           self.target = target
           self.target_unlabeled = target_unlabeled
           self.target_bkg = target_bkg

    def __repr__(self):
        n_source = 0 if self.source is None else len(self.source)
        n_source_bkg = 0 if self.source_bkg is None else len(self.source_bkg)
        n_target = 0 if self.target is None else len(self.target)
        n_target_unlabeled = 0 if self.target_unlabeled is None\
            else len(self.target_unlabeled)
        n_target_bkg = 0 if self.target_bkg is None else len(self.target_bkg)

        return f"Optical Dataset composed of\n\
{n_source} source samples\n\
{n_source_bkg} source background samples\n\
{n_target} target labeled samples\n\
{n_target_unlabeled} target unlabeled samples\n\
{n_target_bkg} target background samples\n"

    def __len__(self):
        return len(self.source) + len(self.target)
            # len(self.source_bkg) +\
            # len(self.target_unlabeled) +\
            # len(self.target_bkg)

    def __getitem__(self, idx):
        """ This fake getitem is needed for sklearn.utils.validation to not treat this object as a scalar
        """
        return None

    def slice(self, fold):
        return OpticalDataset(
            self.source[fold.source],
            self.source_bkg[fold.source_bkg],
            self.target[fold.target],
            self.target_unlabeled[fold.target_unlabeled],
            self.target_bkg[fold.target_bkg])


class OpticalLabels:

    def __init__(self, source=None, target=None):
           self.source = source
           self.target = target

    def __repr__(self):
        n_source = 0 if self.source is None else len(self.source)
        n_target = 0 if self.target is None else len(self.target)
        return f"Optical Dataset labels composed of\n\
{n_source} labels of source samples\n\
{n_target} labels of target samples\n"

    def __len__(self):
        return len(self.source) + len(self.target)

    def __getitem__(self, idx):
        """ This fake getitem is needed for sklearn.utils.validation to not treat this object as a scalar
        """
        return None

    def slice(self, fold):
        return OpticalLabels(
            self.source[fold.source],
            self.target[fold.target])
