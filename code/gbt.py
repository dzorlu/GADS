#!/usr/bin/env python
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

TRAIN_PCT = 0.2     # note this is very low!
def main():

    # generate synthetic binary classification data
    # (name refers to example 10.2 in ESL textbook...see refs below)
    X, y = make_hastie_10_2()

    # perform train/test split (no need to shuffle)
    split_pt = int(TRAIN_PCT * len(X))
    X_train, X_test = X[:split_pt], X[split_pt:]
    y_train, y_test = y[:split_pt], y[split_pt:]

    # single dec stump
    stump_clf = DecisionTreeClassifier(
        max_depth=1)
    stump_clf.fit(X_train, y_train)
    stump_score = round(stump_clf.score(X_test, y_test), 3)
    print 'decision stump acc = {}\t(max_depth = 1)'.format(stump_score)

    # single dec tree (max_depth=3)
    tree_clf = DecisionTreeClassifier(max_depth=5)
    tree_clf.fit(X_train, y_train)
    tree_score = round(tree_clf.score(X_test, y_test), 3)
    print 'decision tree acc = {}\t(max_depth = 5)\n'.format(tree_score)

    # gbt: a powerful ensemble technique
    gbt_scores = list()
    for k in (10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500):
        print 'fitting gbt for n_estimators = {}...'.format(k)

        gbt_clf = GradientBoostingClassifier(
            n_estimators=k,         # number of weak learners for this iteration
            max_depth=1,            # weak learners are dec stumps
            learning_rate=1.0)      # regularization (shrinkage) hyperparam

        gbt_clf.fit(X_train, y_train)
        gbt_scores.append(round(gbt_clf.score(X_test, y_test), 3))

    print '\ngbt accuracy =\n{}'.format(gbt_scores)

if __name__ == '__main__':
    main()

# refs:
# http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting
# http://en.wikipedia.org/wiki/Gradient_boosting
# http://statweb.stanford.edu/~tibs/ElemStatLearn/
