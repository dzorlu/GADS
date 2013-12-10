#!/usr/bin/env python
import pandas as pd

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
    make_scorer, zero_one_loss)
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC

ABALONE_FILE = 'abalone.csv'
STANDARDIZE = False
TRAIN_PCT = 0.30

def get_abalone19():
    """Loads abalone dataset, maps gender feature to binary features, adds
    new label to create abalone19 imbalanced binary classification dataset."""
    raw_data = pd.read_csv(ABALONE_FILE, sep=',')
    genders = list(raw_data.ix[:, 'gender'])
    cts_data = raw_data.drop(labels='gender', axis=1)

    # initialize & fit preprocesser
    lbz = LabelBinarizer()
    lbz.fit(genders)

    # encode categorical var
    encoded_genders = pd.DataFrame(lbz.transform(genders))
    encoded_genders.columns = ['gender_' + k for k in lbz.classes_]

    # recombine encoded data & return
    new_data = pd.concat(objs=[encoded_genders, cts_data], axis=1)
    new_data['label'] = raw_data['rings'].map(
        lambda k: 1 if k > 10 else 0)               # binary clf task
    new_data = new_data.drop('rings', axis=1)

    # standardize cts features
    if STANDARDIZE:
        for col in new_data.ix[:, 3:-1]:
            mean = new_data[col].mean()
            std = new_data[col].std()
            new_data[col] = new_data[col].map(lambda k: (k - mean) / float(std))

    pos_recs = new_data['label'].sum()
    print 'total pos class pct = {} %\n'.format(
        round(100 * pos_recs / float(len(new_data)), 3))

    return new_data

def grid_fit(data):
    X, y = data.ix[:, :-1], data.ix[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size = int(TRAIN_PCT * len(X)))

    # (stochastic) gradient boosted trees
    gbt = GradientBoostingClassifier(subsample=0.8, min_samples_leaf=50,
        min_samples_split=20)

    gbt_params = {'max_depth': [1, 2, 3, 4],
        'n_estimators': [10, 20, 50],
        'learning_rate': [0.1, 0.5, 1.0]}

    # linear svc
    svc = LinearSVC(dual=False)
    svc_params = {'C': [10 ** -k for k in range(5)],
        'class_weight': [{1: 1}],
        'loss': ['l2'],
        'penalty': ['l1']}
        
    # NOTE specify classifier object here
    clf = gbt
    param_grid = gbt_params

    zero_one_loss_obj = make_scorer(zero_one_loss,
        normalize=False,
        greater_is_better=False)

    print 'thinking...'
    grid_results = GridSearchCV(clf, param_grid,
        scoring='roc_auc',
        # scoring=zero_one_loss_obj,
        cv=StratifiedKFold(y_train, n_folds=3),
        # cv=10,
        verbose=1)

    grid_results.fit(X_train, y_train)

    print '\ngenlzn errors:'
    for params, mean_score, all_scores in grid_results.grid_scores_:
        print '{}\t{}\t(+/-) {}'.format(
            params,
            round(mean_score, 3),
            round(all_scores.std() / 2, 3))

    print '\nbest model:'
    print '{}\t{}'.format(grid_results.best_params_,
        round(grid_results.best_score_, 3))

    print '\nclassification report:\n'
    print classification_report(y_test, grid_results.predict(X_test))

    print 'confusion matrix ({} total test recs, {} positive)'.format(
        len(y_test), sum(y_test))
    print confusion_matrix(y_test, grid_results.predict(X_test),
        labels=[1, 0])
    print

if __name__ == '__main__':
    abalone19 = get_abalone19()
    grid_fit(abalone19)
