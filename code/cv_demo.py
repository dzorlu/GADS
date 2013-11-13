#!/usr/bin/env python

import numpy as np
from sklearn import cross_validation as cv

def print_data(features, labels):
    print '=' * 40
    print 'FULL DATASET'
    print '=' * 40

    print 'features = \n{0}\n'.format(features)
    print 'labels = \n{0}\n'.format(labels)

def holdout(features, labels, train_pct=0.7):

    # perform train/test split (via sklearn)
    tt_split_output = cv.train_test_split(features, labels,
        train_size=train_pct)

    # extract output via "tuple unpacking"
    # (note that the commas define the LHS as a tuple, even w/o parentheses!)
    features_train, features_test, labels_train, labels_test = tt_split_output

    print 'TRAINING SET ='
    print '{0}\t\t{1}'.format(features_train, labels_train)
   
    print '\nTEST SET = '
    print '{0}\t\t{1}'.format(features_test, labels_test)

def kfolds(features, labels, k=4):

    # this is our cv iterator (from sklearn)...note that it doesn't depend on
    # the dataset! it's just permuting indexes based on the params we give it
    kf = cv.KFold(n=len(features), n_folds=k, shuffle=True)

    # iterate over kf (and maintain loop counter in i w/ enumerate keyword)
    for i, (train_index, test_index) in enumerate(kf):
        print '=' * 40
        print 'FOLD {0}'.format(i+1)
        print '=' * 40, '\n'

        print 'TRAINING SET ='
        print '{0}\t\t{1}'.format(features[train_index], labels[train_index])

        print '\nTEST SET ='
        print '{0}\t\t{1}\n'.format(features[test_index], labels[test_index])

if __name__ == '__main__':

    # pretend features
    x = np.array([['x01', 'x02'], ['x11', 'x12'], ['x21', 'x22'],
        ['x31', 'x32'], ['x41', 'x42'], ['x51', 'x52'], ['x61', 'x62'],
        ['x71', 'x72'], ['x81', 'x82'], ['x91', 'x92']])

    # pretend labels
    y = np.array(['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9'])

    # print_data(x, y)
    holdout(x, y)
    # kfolds(x, y)
