import os
import copy

from .dataset import OpticalDataset, OpticalLabels
from .cv import CVFold
from rampwf.utils.importing import import_module_from_source


class FeatureExtractor(object):
    def __init__(self, workflow_element_names=['feature_extractor']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X, y, fold=None):
        if fold is None:
            fold = CVFold._none_slice()
        feature_extractor = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=False)
        fe = feature_extractor.FeatureExtractor()
        return fe

    def test_submission(self, trained_model, X):
        return trained_model.transform(X)


class Classifier(object):
    def __init__(self, workflow_element_names=['classifier']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X, y, fold=None):
        classifier = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=False)
        clf = classifier.Classifier()
        clf.fit(
            X.source, X.source_bkg, X.target, X.target_unlabeled,
            X.target_bkg, y.source, y.target)
        return clf

    def test_submission(self, trained_model, X):
        clf = trained_model
        y_proba = clf.predict_proba(X.target, X.target_bkg)
        return y_proba


class FeatureExtractorClassifier(object):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        self.element_names = workflow_element_names
        self.feature_extractor_workflow = FeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = Classifier([self.element_names[1]])

    def train_submission(self, module_path, X, y, fold=None):
        if fold is None:
            fold = CVFold._none_slice()
        # We need to make a copy here
        X = OpticalDataset(
            X.source[fold.source],
            X.source_bkg[fold.source_bkg],
            X.target[fold.target],
            X.target_unlabeled[fold.target_unlabeled],
            X.target_bkg[fold.target_bkg])
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X, y, fold)
        X.source = self.feature_extractor_workflow.test_submission(
            fe, X.source)
        X.source_bkg = self.feature_extractor_workflow.test_submission(
            fe, X.source_bkg)
        X.target = self.feature_extractor_workflow.test_submission(
            fe, X.target)
        X.target_unlabeled = self.feature_extractor_workflow.test_submission(
            fe, X.target_unlabeled)
        X.target_bkg = self.feature_extractor_workflow.test_submission(
            fe, X.target_bkg)
        y = OpticalLabels(y.source[fold.source], y.target[fold.target])
        clf = self.classifier_workflow.train_submission(module_path, X, y)
        return fe, clf

    def test_submission(self, trained_model, X):
        fe, clf = trained_model
        X_copy = copy.deepcopy(X)
        X_copy.target = self.feature_extractor_workflow.test_submission(
            fe, X.target)
        X_copy.target_bkg = self.feature_extractor_workflow.test_submission(
            fe, X.target_bkg)
        y_proba = self.classifier_workflow.test_submission(clf, X_copy)
        return y_proba
