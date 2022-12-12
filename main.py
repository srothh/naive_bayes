import timeit

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import arff
from sklearn.model_selection import train_test_split as tts, train_test_split, cross_val_score
from sklearn.naive_bayes import CategoricalNB
from NaiveBayes import NaiveBayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def labelEncode(encoder, frame, column):
    return encoder.fit_transform(frame[column])


# does pre-processing(missing values, label encoding) for open_payments dataset
def load_open_payments():
    open_payments = None
    # load open_payments data
    with open('data/dataset.arff') as op:
        data = arff.load(op)
        cols = [a for a, b in data['attributes']]
        open_payments = pd.DataFrame(data['data'], columns=cols)
    # label encoding
    le = preprocessing.LabelEncoder()
    for column in open_payments.columns:
        open_payments[column] = labelEncode(le, open_payments, column)
    return open_payments


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    open_payments = load_open_payments()
    nbc = NaiveBayes()
    categorical_nbc = CategoricalNB()
    y = open_payments["status"]
    X = open_payments.drop(columns=['status'])
    breast_cancer = pd.read_csv('data/breast-cancer-diagnostic.shuf.lrn.csv')
    rfc = RandomForestClassifier()
    gaussian_nbc = GaussianNB()
    start = timeit.default_timer()
    cv = cross_val_score(nbc, X, y, cv=5, scoring="accuracy")
    stop = timeit.default_timer()
    print("Open Payments")
    print("Cross-Validation Mean for own Naive Bayes:", cv.mean())
    print("Cross-Validation STD for own Naive Bayes", cv.std())
    print("Runtime for own Naive Bayes:", stop - start)
    start = timeit.default_timer()
    cv = cross_val_score(categorical_nbc, X, y, cv=5, scoring="accuracy")
    stop = timeit.default_timer()
    print("Cross-Validation Mean for reference Naive Bayes:", cv.mean())
    print("Cross-Validation STD for reference Naive Bayes", cv.std())
    print("Runtime for reference Naive Bayes", stop - start)
    start = timeit.default_timer()
    cv = cross_val_score(rfc, X, y, cv=5, scoring="accuracy")
    stop = timeit.default_timer()
    print("Cross-Validation Mean for Random Forest:", cv.mean())
    print("Cross-Validation STD for Random Forest", cv.std())
    print("Runtime for Random Forest", stop - start)
    y = breast_cancer['class']
    X = breast_cancer.drop(['class', 'ID'], axis=1)
    print("Breast Cancer")
    start = timeit.default_timer()
    cv = cross_val_score(nbc, X, y, cv=5, scoring="accuracy")
    stop = timeit.default_timer()
    print("Cross-Validation Mean for own Naive Bayes:", cv.mean())
    print("Cross-Validation STD for own Naive Bayes", cv.std())
    print("Runtime for own Naive Bayes:", stop - start)
    start = timeit.default_timer()
    cv = cross_val_score(gaussian_nbc, X, y, cv=5, scoring="accuracy")
    stop = timeit.default_timer()
    print("Cross-Validation Mean for reference Naive Bayes:", cv.mean())
    print("Cross-Validation STD for reference Naive Bayes", cv.std())
    print("Runtime for own Naive Bayes", stop - start)
    start = timeit.default_timer()
    cv = cross_val_score(rfc, X, y, cv=5, scoring="accuracy")
    stop = timeit.default_timer()
    print("Cross-Validation Mean for Random Forest:", cv.mean())
    print("Cross-Validation STD for Random Forest", cv.std())
    print("Runtime for own Naive Bayes", stop - start)
