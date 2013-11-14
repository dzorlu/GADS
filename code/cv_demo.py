#!/usr/bin/env python

import numpy as np
import pandas as pd
import pylab as pl      # note this is part of matplotlib

from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve, auc

TRAIN_PCT = 0.7
NUM_FOLDS = 4
INPUT_FILE = 'beer.txt'

def print_data(features, labels):
    print '=' * 40
    print 'FULL DATASET'
    print '=' * 40

    print 'features = \n{0}\n'.format(features)
    print 'labels = \n{0}\n'.format(labels)

def holdout(features, labels):

    # perform train/test split (via sklearn)
    tt_split_output = cv.train_test_split(features, labels,
        train_size=TRAIN_PCT)

    # extract output via "tuple unpacking"
    # (note that the commas define the LHS as a tuple, even w/o parentheses!)
    features_train, features_test, labels_train, labels_test = tt_split_output

    print '=' * 40
    print 'HOLDOUT'
    print '=' * 40

    print 'TRAINING SET ='
    print '{0}\t\t{1}'.format(features_train, labels_train)
   
    print '\nTEST SET = '
    print '{0}\t\t{1}'.format(features_test, labels_test)

def kfolds(features, labels):

    # this is our cv iterator (from sklearn)...note that it doesn't depend on
    # the dataset! it's just permuting indexes based on the params we give it
    kf = cv.KFold(n=len(features), n_folds=NUM_FOLDS, shuffle=True)

    # iterate over kf (and maintain loop counter in i w/ enumerate keyword)
    for i, (train_index, test_index) in enumerate(kf):
        print '=' * 40
        print 'FOLD {0}'.format(i+1)
        print '=' * 40, '\n'

        print 'TRAINING SET ='
        print '{0}\t\t{1}'.format(features[train_index], labels[train_index])

        print '\nTEST SET ='
        print '{0}\t\t{1}\n'.format(features[test_index], labels[test_index])

def roc_it(input_file=INPUT_FILE):

    beer = pd.read_csv(input_file, delimiter='\t').dropna()

    # add class label for top half / bottom half
    midpt = int(len(beer) / 2)
    beer['label'] = beer['Rank'].map(lambda k: 1 if k <= midpt else 0)

    # drop categorical columns
    features = beer[['ABV', 'Reviews']]
    labels = beer['label']

    # create cv iterator (note: train pct is set implicitly by number of folds)
    num_recs = len(beer)
    kf = cv.KFold(n=num_recs, n_folds=NUM_FOLDS, shuffle=True)

    # initialize results sets
    all_fprs, all_tprs, all_aucs = (np.zeros(NUM_FOLDS), np.zeros(NUM_FOLDS),
        np.zeros(NUM_FOLDS))

    for i, (train_index, test_index) in enumerate(kf):

        # initialize & train model
        model = LR()

        # debug!
        train_features = features.loc[train_index].dropna()
        train_labels = labels.loc[train_index].dropna()

        test_features = features.loc[test_index].dropna()
        test_labels = labels.loc[test_index].dropna()

        model.fit(train_features, train_labels)

        # predict labels for test features
        pred_labels = model.predict(test_features)

        # calculate ROC/AUC
        fpr, tpr, thresholds = roc_curve(test_labels, pred_labels, pos_label=1)
        roc_auc = auc(fpr, tpr)

        print '\nfpr = {0}'.format(fpr)
        print 'tpr = {0}'.format(tpr)
        print 'auc = {0}'.format(roc_auc)

        all_fprs[i] = fpr[1]
        all_tprs[i] = tpr[1]
        all_aucs[i] = roc_auc

    print '\nall_fprs = {0}'.format(all_fprs)
    print 'all_tprs = {0}'.format(all_tprs)
    print 'all_aucs = {0}'.format(all_aucs)

    # plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show() 

if __name__ == '__main__':

    # pretend features
    x = np.array([['x01', 'x02'], ['x11', 'x12'], ['x21', 'x22'],
        ['x31', 'x32'], ['x41', 'x42'], ['x51', 'x52'], ['x61', 'x62'],
        ['x71', 'x72'], ['x81', 'x82'], ['x91', 'x92']])

    # pretend labels
    y = np.array(['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9'])

    # print_data(x, y)
    # holdout(x, y)
    # kfolds(x, y)

    roc_it()
