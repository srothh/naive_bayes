import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import typing

from sklearn.base import BaseEstimator, ClassifierMixin

#BaseEstimator and ClassifierMixin only for sklearn cross validation functionality
class NaiveBayes(BaseEstimator, ClassifierMixin):
    likelihoods = {}
    fitted = False
    classes = []
    length = 0
    priors = {}
    col_numeric = ['int64', 'float64']

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.classes = y
        df = X.join(y)
        self.length = len(y)
        class_counts = self.classes.value_counts()
        for class_name in self.classes.unique():
            count = class_counts[class_name]
            self.priors[class_name] = count / self.length  # get prior probability for each class and store in a dict
            for col, typ in zip(X.columns, X.dtypes):
                if typ in self.col_numeric:
                    class_col = X.loc[df[y.name] == class_name][col]
                    mean = class_col.mean()
                    sigma = class_col.std()
                    self.likelihoods[(col, 'numeric', class_name)] = \
                        lambda p, mean=mean, sigma=sigma: np.exp(-((p - mean) ** 2) / (2 * (sigma ** 2))) / (
                                np.sqrt(2 * np.pi) * sigma)
                else:
                    for val, cnt in (df.loc[df[y.name] == class_name])[col].value_counts().items():
                        self.likelihoods[(col, val, class_name)] = lambda p, cnt=cnt, count=count: (cnt + 1) / count
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame):
        if not self.fitted:
            print("Fit before predicting")
            return []

        res_pred_rows = []

        for _, row in X.iterrows():
            guess = None
            class_val = float('-inf')
            for c in self.classes.unique():
                p = self.priors[c]
                for col, typ in zip(X.columns, X.dtypes):
                    val = self.likelihoods.get((col, 'numeric' if typ in self.col_numeric else row[col], c))
                    p *= val(p=row[col]) if val is not None else 1 / self.length
                if p > class_val:
                    class_val = p
                    guess = c
            res_pred_rows.append(guess)
        return res_pred_rows
