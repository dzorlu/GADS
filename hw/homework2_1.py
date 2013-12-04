#!/usr/bin/env python
import numpy as np
import pylab as pl
import pandas as pd


from sklearn import cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt

N_FOLDS = 10
TRAIN_PCT = 0.7
STEPS  = 20
K = 21 #After Running CV on KNN

DATA = 'wine.txt'


def full_knn(data=DATA, steps = STEPS, n_folds = N_FOLDS):
    """Perform knn classification on wine dataset across different k's,
     shows results."""

    #Labels/Features
    wine = pd.read_csv(data,header=None).dropna()
    labels = wine[0]
    features = wine.drop(0,axis=1)
    # ranges of k. Odd Integers
    k_range = range(1,steps*2,2)
    # perform cross validation
    kf = cv.KFold(n=len(features), n_folds=n_folds, shuffle=True)
    #Accuracy 
    acc = np.zeros((len(k_range),n_folds))
    #Iterate over different k's. 
    for idx, k in enumerate(k_range):
        # iterate over kf (and maintain loop counter in i w/ enumerate keyword)
        for i, (train_index, test_index) in enumerate(kf):
            #Train / Test Split
            features_train = features.iloc[train_index]
            features_test = features.iloc[test_index]
            
            labels_train = labels.iloc[train_index]
            labels_test = labels.iloc[test_index]
            
            # initialize KNN model, perform fit
            clf = KNN(n_neighbors=k)
            clf.fit(features_train, labels_train)
            # get accuracy (predictions made internally)
            acc[idx,i] = clf.score(features_test, labels_test)

        #Report
        print 'k = {0}'.format(k)
        print 'accuracy = {0} %\n'.format(round(100 * np.mean(acc[idx,:]), 2))
    #Plot
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(k_range,np.mean(acc,1))
    plt.show()
    plt.savefig('Accuracy Across Differnt Ks')

def model_cv(data=DATA,k=K,n_folds=N_FOLDS):
    #Labels/Features
    wine = pd.read_csv(data,header=None).dropna()
    labels = wine[0]
    features = wine.drop(0,axis=1)
    #Initialize arrays
    acc_knn = np.zeros(n_folds)
    acc_lr = np.zeros(n_folds)    
    # initialize KNN model, perform fit
    clf_knn = KNN(n_neighbors=k)
    clf_lr = LR()
    # get accuracy (predictions made internally)
    score_knn = cv.cross_val_score(clf_knn, \
        features, labels, cv=n_folds).mean()  
    score_lr = cv.cross_val_score(clf_lr, \
        features, labels, cv=n_folds).mean() 
    print 'cross-validation score using KNN is {0}'.format(score_knn)
    print 'cross-validation score using LR is {0}'.format(score_lr)



if __name__ == '__main__':
    full_knn()
    model_cv()


