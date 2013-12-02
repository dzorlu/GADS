#!/usr/bin/env python
import StringIO

import pandas as pd
import pydot

from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing as preproc
from sklearn import tree

INPUT_FILE = 'abalone.csv'
TRAIN_PCT = 0.8
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
    bz = preproc.Binarizer()
    return new_data

def run_model(data):
    """Do some label bucketing, print model output."""
    features = data.ix[:, :-1]

    # more categories <--> less accuracy
    # labels = data.ix[:, -1]
    # labels = data.ix[:, -1].map(lambda k: 1 if k > 10 else 0)
    labels = data.ix[:, -1].map(lambda k: int(k / 5))     # bucketing trick

    # initialize model (w/ params)
    clf = tree.DecisionTreeClassifier(
        min_samples_leaf=50)
        # max_depth=3)

    # print cross-validated accuracy results
    cv_results = cross_val_score(clf, features, labels, cv=5)
    print 'cv_results = {}'.format(cv_results)
    print 'avg accuracy = {}'.format(cv_results.mean())

    # show feature importances (can be useful for feature selection)
    clf.fit(features, labels)
    print '\nfeature importances = \n{}'.format(clf.feature_importances_)

    # create dec tree graph as pdf
    create_pdf(clf)

def create_pdf(clf):
    """Save dec tree graph as pdf."""
    dot_data = StringIO.StringIO() 
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('abalone.pdf')

if __name__ == '__main__':
    abalone_data = get_data()
    run_model(abalone_data)

# NOTE here are the dependencies for exporting the dec tree image as a pdf
# 1) graphviz -- this is an application, not a python package...it can be
#    installed on OSX via homebrew by typing "brew install graphviz"
# 2) pydot ("pip install pydot")
# 3) pyparsing -- make sure you install version 1.5.7! ("pip install pyparsing==1.5.7")
