#!/usr/bin/env python
import numpy as np
import pylab as pl

from matplotlib.colors import ListedColormap

from sklearn import cross_validation as cv
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.preprocessing import scale

NUM_NBRS = 3
MESH_SIZE = 0.02

COLORS_1 = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
COLORS_2 = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

TRAIN_PCT = 0.7

def plot_dby(iris):
    """Performs knn classification on projected iris dataset, plots results as
    well as decision boundaries."""

    # project features into 2-dim space (for viz purposes)
    # NOTE "projection" just means that we're dropping the other features...this is
    #      not the same thing as "feature selection" (which requires more care)
    #      or "dimensionality reduction" (which requires more math)
    X = iris.data[:, :2]
    y = iris.target
    
    # initialize & fit knn model (k = 15)
    clf = knn(n_neighbors=NUM_NBRS)
    clf.fit(X, y)

    # create x, y mesh to plot decision boundaries
    x_min = -1 + X[:, 0].min()
    y_min = -1 + X[:, 1].min()

    x_max = 1 + X[:, 0].max()
    y_max = 1 + X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_SIZE),
        np.arange(y_min, y_max, MESH_SIZE))

    # create predictions & reshape to fit mesh
    preds = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    preds = preds.reshape(xx.shape)

    # plot prediction results
    pl.figure()
    pl.pcolormesh(xx, yy, preds, cmap=COLORS_1)

    # plot training examples
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=COLORS_2)

    # set additional plot parameters
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.title('knn classification of iris dataset (k = {0})'.format(NUM_NBRS))
    
    pl.show()

def full_knn(iris, num_features=4):
    """Perform knn classification on iris dataset using given number of
    feature dimensions (default = 4), shows results."""

    # perform projection
    iris.data = iris.data[:,: num_features]

    # screw up scaling! (knn can be sensitive to feature scaling)
    # iris.data[:, :1] *= 100000000

    # perform train/test split
    tts = cv.train_test_split(iris.data, iris.target, train_size=TRAIN_PCT)
    train_features, test_features, train_labels, test_labels = tts

    # initialize model, perform fit
    clf = knn(n_neighbors=NUM_NBRS)
    clf.fit(train_features, train_labels)

    # get accuracy (predictions made internally)
    acc = clf.score(test_features, test_labels)

    # get conf matrix (requires predicted labels)
    predicted_labels = clf.predict(test_features)
    cm = confusion_matrix(test_labels, predicted_labels)

    print 'k = {0}'.format(NUM_NBRS)
    print 'num_features = {0}'.format(num_features)
    print 'accuracy = {0} %\n'.format(round(100 * acc, 2))
    print 'confusion matrix:\n', cm, '\n'

    # NOTE knn can be quite effective for the iris dataset, as you can see
    # what difficulties might arise with
    #      a)  a larger dataset (more rows)?
    #      b)  higher dimensional data (more columns)?
    # how could you control for these difficulties?

if __name__ == '__main__':
    dataset = load_iris()

    plot_dby(dataset)
    # full_knn(dataset)

# adapted from example at
# http://scikit-learn.org/stable/modules/neighbors.html
# (read the docs!)
