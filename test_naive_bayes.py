import pandas as pd
from naive_bayes import NaiveBayes



def test_categorical():

    train_data = pd.DataFrame(
        {
            'street': ['dry','wet','dry', 'dry', 'wet', 'wet'],
            'sky': ['sunny','sunny','cloudy', 'cloudy', 'sunny', 'sunny'],
            'target': ['F', 'F', 'T', 'F', 'C', 'C']
        }
    )
    clf = NaiveBayes().fit(train_data.drop('target', axis=1), train_data['target'])

    # Predict
    pred = clf.predict(pd.DataFrame({
        'street': ['dry', 'wet'],
        'sky': ['cloudy', 'sunny']
    }))
    assert pred == ['T', 'C']


def test_numeric():

    train_data = pd.DataFrame(
        {
            'street': [10,      11,     8,      20,    16,     120,    210],
            'sky':    [1,       2,      0,      2,     5,      90,     70],
            'target': ['F',    'F',    'F',    'T',    'T',    'C',     'C']
        }
    )
    clf = NaiveBayes().fit(train_data.drop('target', axis=1), train_data['target'])

    # Predict
    test_df = pd.DataFrame({
        'street':   [13, 200, 14, 40],
        'sky':      [1,  80,  2,  -10]
    })
    pred = clf.predict(test_df)

    assert pred == ['F', 'C', 'T', 'C']
