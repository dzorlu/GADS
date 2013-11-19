#!/usr/bin/env python

import numpy as np
import pandas as pd
import pylab as pl      # note this is part of matplotlib

from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFECV

NUM_FOLDS = 5
TRAIN_FEATURES = 'logit-train.csv'
TEST_FEATURES = 'logit-test.csv'
METHOD = 'features'


def roc_it(train_features=TRAIN_FEATURES,test_features = TEST_FEATURES,method=METHOD):

    train_features = pd.read_csv(train_features).dropna()
    test_features = pd.read_csv(test_features).dropna()

    train_labels = train_features['heartdisease::category|0|1']
    train_features = train_features.drop('heartdisease::category|0|1',axis=1)

    test_labels = test_features['heartdisease::category|0|1']
    test_features = test_features.drop('heartdisease::category|0|1',axis=1)


    ##COLUMN SELECTION
    if method in 'features':
        print 'processing features'
        # create cv iterator (note: train pct is set implicitly by number of folds)
        num_features = train_features.columns.size
        kf = cv.KFold(n=num_features, n_folds=NUM_FOLDS, shuffle=True)
        # initialize results sets
        all_fprs, all_tprs, all_aucs = (np.zeros(NUM_FOLDS), np.zeros(NUM_FOLDS),
            np.zeros(NUM_FOLDS))
        #RFECV
        selector = RFECV(LR(), step=1, cv=NUM_FOLDS)
        selector = selector.fit(train_features, train_labels)
        print selector.ranking_

        for i, (index, test_index) in enumerate(kf):

            # initialize & train model
            model = LR()

            model.fit(train_features.iloc[:,index], train_labels)
            # predict labels for test features
            pred_labels = model.predict(test_features.iloc[:,index])

            # calculate ROC/AUC
            fpr, tpr, thresholds = roc_curve(test_labels, pred_labels, pos_label=1)
            roc_auc = auc(fpr, tpr)
            accuracy = model.score(test_features.iloc[:,index], test_labels)
            print 'coefficients ', model.coef_
            print 'accuracy is ', accuracy
            print '................'
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
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % all_aucs.mean())
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic example - %s' %(method))
        pl.legend(loc="lower right")
        pl.show()
        pl.savefig('pic_features.png')

    elif method in 'records':
        print 'processing records'
        #RECORDS
        num_recs = len(train_features)
        kf = cv.KFold(n=num_recs, n_folds=NUM_FOLDS, shuffle=True)
        # initialize results sets
        all_fprs, all_tprs, all_aucs = (np.zeros(NUM_FOLDS), np.zeros(NUM_FOLDS),
            np.zeros(NUM_FOLDS))
        # +1 for the intercept
        coef = np.zeros(shape=(NUM_FOLDS,train_features.columns.size+1))

        for i, (train_index, test_index) in enumerate(kf):

            # initialize & train model
            model = LR()

            #Temp Feature Set
            temp_train_features = train_features.loc[train_index].dropna()
            temp_train_labels = train_labels.loc[train_index].dropna()

            temp_test_features = test_features.loc[test_index].dropna()
            temp_test_labels = test_labels.loc[test_index].dropna()

            model.fit(temp_train_features, temp_train_labels)
            coef[i,:] = model.raw_coef_

            # predict labels for test features
            pred_labels = model.predict(temp_test_features)

            # calculate ROC/AUC
            fpr, tpr, thresholds = roc_curve(temp_test_labels, pred_labels, pos_label=1)
            roc_auc = auc(fpr, tpr)
            #Accuracy
            accuracy = model.score(temp_test_features, temp_test_labels)
            print 'coefficiencts ', model.coef_
            print 'accuracy is ', accuracy
            print '................'
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
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % all_aucs.mean())
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic example - %s' %(method))
        pl.legend(loc="lower right")
        pl.show()
        pl.savefig('pic_records.png')
        #Mean of Coefficients from folds
        coef_mean = coef.mean(0)
        return(coef_mean)



if __name__ == '__main__':

    roc_it()
