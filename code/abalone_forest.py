#!/usr/bin/env python
import StringIO

import pandas as pd

from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing as preproc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

INPUT_FILE = 'abalone.csv'
N_RECS = 500

N_FOLDS = 5
MAX_DEPTH = 3
N_TREES = 100
N_JOBS = -1     # auto-magically detect number of cores

def get_data():
    """Load data from file & preprocess categorical variable."""
    
    # load data & separate categorical var
    raw_data = pd.read_csv(INPUT_FILE, sep=',')
    genders = list(raw_data.ix[:, 'gender'])
    cts_data = raw_data.drop(labels='gender', axis=1)

    # initialize & fit preprocesser
    lbz = preproc.LabelBinarizer()
    lbz.fit(genders)

    # encode categorical var
    encoded_genders = pd.DataFrame(lbz.transform(genders))
    encoded_genders.columns = ['gender_' + k for k in lbz.classes_]

    # recombine encoded data & return
    new_data = pd.concat(objs=[encoded_genders, cts_data], axis=1)

    print 'num recs = {}'.format(N_RECS)
    return new_data.ix[: N_RECS, :]     # use fewer recs to emphasize ensemble effectiveness

def run_model(data):
    """Do some label bucketing, print model output."""
    features = data.ix[:, :-1]

    # more categories <--> less accuracy
    # labels = data.ix[:, -1].map(lambda k: 1 if k > 10 else 0)
    labels = data.ix[:, -1].map(lambda k: int(k / 5))     # bucketing trick
    print 'num classes = {}\n'.format(len(set(labels)))

    # weak (base) classifier
    print 'fitting weak classifier...'
    weak_clf = DecisionTreeClassifier(max_depth=MAX_DEPTH)

    weak_cv_results = cross_val_score(weak_clf, features, labels,
        cv=N_FOLDS)
    print 'weak_cv_results = {}'.format(weak_cv_results)
    print 'avg accuracy = {}\n'.format(weak_cv_results.mean())
    
    # strong (ensemble) classifier
    print 'fitting strong classifier...'
    strong_clf = RandomForestClassifier(
        max_depth=MAX_DEPTH,
        n_estimators=N_TREES,
        n_jobs=N_JOBS)

    strong_cv_results = cross_val_score(strong_clf, features, labels,
        cv=N_FOLDS)
    print 'strong_cv_results = {}'.format(strong_cv_results)
    print 'avg accuracy = {}'.format(strong_cv_results.mean())

if __name__ == '__main__':
    abalone_data = get_data()
    run_model(abalone_data)
