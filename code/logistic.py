#!/usr/bin/env python
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix

INPUT_FILE = 'beer.txt'
TRAIN_PCT = 0.7

def add_dummys(data):
    """Create dummy variables for some generalizations of 'Type' column
    (note: this is a simplistic example of clustering in 1d)"""

    data['type_Stout'] = data['Type'].map(lambda k: 1 if 'Stout' in k else 0)
    data['type_IPA'] = data['Type'].map(lambda k: 1 if 'IPA' in k else 0)
    data['type_Ale'] = data['Type'].map(lambda k: 1 if 'Ale' in k else 0)

    # note "type_Other" is implicitly encoded as a 0 for each of these new
    # features (in general you need k-1 dummy vars to encode k categorical vars)

    return data

def preprocess_data(input_file=INPUT_FILE, simple=1):
    """Load data from txt file, perform preprocessing and return df."""

    # load dataset, drop na's
    beer = pd.read_csv(input_file, delimiter='\t').dropna()
    
    # add column for class labels (1 for top half, 0 for bottom half)
    midpt = int(len(beer) / 2)
    beer['label'] = beer['Rank'].map(lambda k: 1 if k <= midpt else 0)

    # notice that there's a lot going on in the above line!
    # (buzzwords: map, anonymous function, ternary operator...look these up!)

    # for the sake of simplicity, our first model fit will use only the
    # numerical features
    if simple:
        return beer[['ABV', 'Reviews', 'label']]

    # add another feature for 'group'
    else:
        beer = add_dummys(beer)
        return beer[['ABV', 'Reviews', 'label',
            'type_Stout', 'type_IPA', 'type_Ale']]

    # note: pandas is based on numpy, which relies heavily on such "functional"
    # ("vectorized") techniques as the map function used here (like matlab, R)

def run_model(data):
    """Perform train/test split, fit model and output results."""

    # shuffle dataset
    data = data.reindex(np.random.permutation(data.index))

    # perform train/test split (more about this next lecture!)
    # btw, scikit will do this for you...but it's good to see it done by hand
    split_pt = int(TRAIN_PCT * len(data))

    train_x = data[:split_pt].drop('label', 1)      # training set features
    train_y = data[:split_pt].label                 # training set target

    test_x = data[split_pt:].drop('label', 1)       # test set features
    test_y = data[split_pt:].label                  # test set target

    # initialize model & perform fit
    model = LR()                        # model is an "instance" of the class LR
    model.fit(train_x, train_y)         # perform model fit ("in place")

    # get model outputs
    inputs = map(str, train_x.columns.format())
    coeffs = model.coef_[0]
    accuracy = model.score(test_x, test_y)

    predicted_y = model.predict(test_x)
    cm = confusion_matrix(test_y, predicted_y)

    print 'inputs = {0}'.format(inputs)
    print 'coeffs = {0}'.format(coeffs)
    print 'accuracy = {0}'.format(accuracy)     # mean 0/1 loss
    print 'confusion matrix:\n', cm, '\n'

if __name__ == '__main__':
    beer = preprocess_data(simple=1)
    # beer = preprocess_data()
    run_model(beer)


# CHALLENGE QUESTIONS!
# 1) what do the accuracy results suggest?
# 2) how do your results change if you try to predict the top 10%
#    instead of the top half? what does this suggest?
# 3) if you run this several times, your results may vary widely. how could you
#    stabilize this behavior?
# 4) why should you worry about a train/test split?
# 5) the model object has attributes called "model.penalty" and "model.c". What
#    are these values used for?
