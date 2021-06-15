from sklearn.ensemble import RandomForestClassifier

class Classifier:

    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=2, max_depth=2, random_state=44, n_jobs=-1)

    def fit(self, X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target):
        self.clf.fit(X_source, y_source)

    def predict_proba(self, X_target, X_target_bkg):
        y_proba = self.clf.predict_proba(X_target)
        return y_proba
