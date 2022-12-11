import pandas as pd
from naive_bayes import NaiveBayes



def test_traivial():

    train_data = pd.DataFrame(
        {
            'street': ['dry','wet','dry', 'dry', 'wet', 'wet'],
            'sky': ['sunny','sunny','cloudy', 'cloudy', 'sunny', 'sunny'],
            'target': ['F', 'F', 'T', 'F', 'C', 'C']
        }
    )
    clf = NaiveBayes().fit(train_data.drop('target', axis=1), train_data['target'])

    test_data = {
            'F': (2 + 1)/3 * (1 + 1)/3 * 3/4,
            'T': (1 + 1)/1 * (1 + 1)/1 * 1/4
        }

    # Predict
    pred = clf.predict(pd.DataFrame({
        'street': ['dry', 'wet'],
        'sky': ['cloudy', 'sunny']
    }))
    assert pred == ['T', 'C']
