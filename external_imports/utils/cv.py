from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


class CVFold:

    def __init__(self,
                 source=None,
                 source_bkg=None,
                 target=None,
                 target_bkg=None):
        self.source = source
        self.source_bkg = source_bkg
        self.target = target
        self.target_unlabeled = target
        self.target_bkg = target_bkg

    @classmethod
    def _none_slice(self):
        none_slice = slice(None, None, None)
        return self(
            source=none_slice, source_bkg=none_slice,
            target=none_slice, target_bkg=none_slice)


class TLShuffleSplit:
    """Cross validation in TL task."""

    def __init__(self, n_splits, test_size=None, random_state=43,
            train_size_labeled_target=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size_labeled_target = train_size_labeled_target
        self.random_state = random_state

    def split(self, X, y):
        # Define Shuffle split for every data domain

        source_split = StratifiedShuffleSplit(
            n_splits=self.n_splits, test_size=self.test_size,
            random_state=self.random_state)
        source_bkg_split = ShuffleSplit(
            n_splits=self.n_splits, test_size=self.test_size,
            random_state=self.random_state)
        target_split = StratifiedShuffleSplit(
            n_splits=self.n_splits, train_size=self.train_size_labeled_target,
            random_state=self.random_state)
        target_unlabeled_split = ShuffleSplit(
            n_splits=self.n_splits, test_size=self.test_size,
            random_state=self.random_state)
        target_bkg_split = ShuffleSplit(
            n_splits=self.n_splits, test_size=self.test_size,
            random_state=self.random_state)

        source_split_iter = source_split.split(X.source, y.source)
        target_split_iter = target_split.split(X.target, y.target)
        source_bkg_split_iter = source_bkg_split.split(X.source_bkg)
        target_unlabeled_split_iter = target_unlabeled_split.split(
            X.target_unlabeled)
        target_bkg_split_iter = target_bkg_split.split(X.target_bkg)

        for _ in range(self.n_splits):
            train_fold = CVFold()
            valid_fold = CVFold()
            train_fold.source, valid_fold.source = next(source_split_iter)
            train_fold.target, valid_fold.target = next(target_split_iter)
            train_fold.source_bkg, valid_fold.source_bkg = next(
                source_bkg_split_iter)
            train_fold.target_unlabeled, valid_fold.target_unlabeled = next(
                target_unlabeled_split_iter)
            train_fold.target_bkg, valid_fold.target_bkg = next(
                target_bkg_split_iter)

            yield train_fold, valid_fold
