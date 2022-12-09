from NaiveBayes import NaiveBayes
import pandas as pd


def test_traivial():
    train_data = pd.DataFrame(
        {
            'street': ['dry','wet','dry', 'dry'],
            'sky': ['suny','suny','cloudy', 'cloudy'],
            'target': ['F', 'F', 'T', 'F']
        }
    )
    clf = NaiveBayes().fit(train_data, 'target')

    test_data = {
            'F': 2/3 * 1/3 * 3/4,
            'T': 1/1 * 1/1 * 1/4,
        }
    norm_const = sum([p for p in test_data.values()])
    test_data['F'] /= norm_const
    test_data['T'] /= norm_const

    print(clf.predict(pd.DataFrame({
        'street': ['dry'],
        'sky': ['cloudy']
    })))
    print(test_data)

test_traivial()