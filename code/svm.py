#!/usr/bin/env python
import pandas as pd

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.datasets import make_circles
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

from ggplot import *

TRAIN_PCT = 0.70

def get_circles():
    """Load synthetic concentric circles data from sklearn, transform into
    dataframe & return."""
    circles_data, circles_labels = make_circles(n_samples=1000, noise=0.1)

    circles_data = pd.DataFrame(circles_data)
    circles_data.columns = ['x', 'y']

    circles_labels = pd.DataFrame(circles_labels)
    circles_labels.columns = ['label']

    circles_combined = pd.concat([circles_data, circles_labels], axis=1)

    # draw scatter plot of concentric circles
    g = (ggplot(circles_combined, aes('x', 'y', shape='label'))
        + geom_point(size=40))
    print g

    return circles_combined
    
def run_svm(data):
    """Perform grid search CV on SVM params, display classification results."""
    X, y = data.ix[:, :-1], data.ix[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size = int(TRAIN_PCT * len(X)))

    clf = SVC()

    param_grid = {
        'C': [10 ** k for k in range(-3, 3)],
        # 'C': range(1, 11),
        'kernel': ['linear', 'poly', 'rbf']}
        # 'kernel': ['rbf']}

    grid_results = GridSearchCV(clf, param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(y_train, n_folds=20),
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

    print '\nclassification report:'
    print classification_report(y_test, grid_results.predict(X_test))

    print 'confusion matrix ({} total test recs, {} positive)'.format(
        len(y_test), sum(y_test))
    print confusion_matrix(y_test, grid_results.predict(X_test),
        labels=[1, 0])
    print

if __name__ == '__main__':
    data = get_circles()
    run_svm(data)
