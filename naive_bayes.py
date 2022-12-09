import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import typing

class NaiveBayes:
    likelihoods = {}
    fitted = False
    classes = []
    length = 0
    priors = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.classes = y
        df = X.join(y)
        self.length = len(y)
        class_counts = self.classes.value_counts()
        for class_name in self.classes.unique():
            count = class_counts[class_name]
            self.priors[class_name] = count / self.length  # get prior probability for each class and store in a dict
            for col in X.columns:
                for val, cnt in (df.loc[df[y.name] == class_name])[col].value_counts().items():
                    self.likelihoods[(col, val, class_name)] = (cnt + 1) / count  # offset by 1 for zero frequency problem
        self.fitted = True
        return self

    # TODO probably lot of mistakes (maybe wrong predictions)
    def predict(self, X: pd.DataFrame):
        if not self.fitted:
            print("Fit before predicting")
            return []

        res_pred_rows = []
        for _, row in X.iterrows():
            class_probs = []
            for c in self.classes.unique():
                p = self.priors[c]
                for col in X.columns:
                    p *= self.likelihoods.get(
                            (col, row[col], c),
                            1 / self.length #default occurence is 1
                        )
                class_probs.append((c, p))
            res_pred_rows.append(class_probs)
        return res_pred_rows
