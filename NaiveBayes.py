import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


class NaiveBayes:
    probabilities = {}
    fitted = False
    target_vals = []
    length = 0

    def fit(self, dataset, target):
        outcomes = []
        self.length = len(dataset[target])
        occurrences = dataset[target].value_counts()
        for val in dataset[target].unique():
            self.target_vals.append((val,
                                     occurrences.val))  # should store for each target how often it occurs in format (value, occurrences)
            outcomes.append(pd.DataFrame(dataset[dataset[target] == val]))
        for outcome in outcomes:
            curr_val = outcome[target].iat[0]
            for col in outcome.columns:
                length = len(outcome[col])
                for val, cnt in outcome[col].value_counts().items():
                    self.probabilities[(col, val, curr_val)] = (cnt + 1) / length  # TODO zero frequency problem
        self.fitted = True

    # TODO probably lot of mistakes
    def predict(self, input):
        if not self.fitted:
            print("Fit Classifier before predicting")
            return
        res = []
        normalization = 0
        p = 0
        for val, occ in self.target_vals:
            for row in input.iterrows:  # iterrows very inefficient, optimise with different method
                p = 0
                for col in input.columns:
                    if p == 0:
                        p = self.probabilities[col, input[row][col], val]
                    else:
                        p *= self.probabilities[col, input[row][col], val]
            res.append((val, p))
            normalization += p
        # normalization
        normalized_res = {}
        for x, y in res:
            normalized_res[x] = y / normalization  # might be completely wrong
        return normalized_res
