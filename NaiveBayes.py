import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


class NaiveBayes:
    likelihoods = {}
    fitted = False
    classes = []
    length = 0
    priors = {}

    def fit(self, x, y):
        self.classes = y
        df = x.join(y)
        self.length = len(y)
        for c in self.classes.unique():
            count = self.classes.value_counts()[c]
            self.priors[c] = count / self.length  # get prior probability for each class and store in a dict
            for col in x.columns:
                for val, cnt in (df.loc[df[y.name] == c])[col].value_counts().items():
                    self.likelihoods[(col, val, c)] = (cnt + 1) / count  # offset by 1 for zero frequency problem
        self.fitted = True

    # TODO probably lot of mistakes
    def predict(self, x):
        if not self.fitted:
            print("Fit before predicting")
            return x
        max_p = -1
        current_guess = self.classes[0]
        res = []
        for i, row in x.iterrows():
            for c in self.classes.unique():
                p = self.priors[c]
                for col in x.columns:
                    p *= self.likelihoods.get((col, row[col], c), 1 / self.length) #default occurence is 1
                if p > max_p:
                    max_p = p
                    current_guess = c
            res.append(current_guess)
            max_p = -1
        return res
