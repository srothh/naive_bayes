import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import typing

class NaiveBayes:
    probabilities = {}
    fitted = False
    target_vals = []
    length = 0

    def fit(self, dataset: pd.DataFrame, target: str):
        outcomes = []
        self.length = len(dataset[target])
        occurrences = dataset[target].value_counts()
        for val in dataset[target].unique():
            self.target_vals.append((val,
                                     occurrences[val]))  # should store for each target how often it occurs in format (value, occurrences)
            outcomes.append(pd.DataFrame(dataset[dataset[target] == val]))
        for outcome in outcomes:
            curr_val = outcome[target].iat[0]
            for col in outcome.columns:
                length = len(outcome[col])
                for val, cnt in outcome[col].value_counts().items():
                    self.probabilities[(col, val, curr_val)] = (cnt + 1) / length  # TODO zero frequency problem
        self.fitted = True
        return self

    # TODO probably lot of mistakes
    def predict(self, input: pd.DataFrame):
        if not self.fitted:
            print("Fit Classifier before predicting")
            return
        res = []


        for row in input.iterrows(): # iterrows very inefficient, optimise with different method
            for target, target_count in self.target_vals:
                p = 1
                for col in input.columns:
                    p *= self.probabilities[col, row[1][col], target] / target_count
                p *= target_count / sum([c for _, c in self.target_vals ])
                res.append((target, p))
        # normalization
        normalized_res = {}
        norm_const = sum([p for _, p in res])
        normalized_res = [(target, p / norm_const) for target, p in res]
        return normalized_res