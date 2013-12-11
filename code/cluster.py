#!/usr/bin/env python
from itertools import combinations as combos

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn import preprocessing as preproc

from ggplot import *

WINE_FILE = 'wine.csv'

def iris_clusters():
    iris = load_iris()
    X, y = iris.data, iris.target

    # scale data!
    X = preproc.scale(X)

    kmeans = KMeans(n_clusters=3).fit(X)
    print 'silhouette coeff = {}'.format(
        round(silhouette_score(X, kmeans.labels_, metric='euclidean'), 3))

    iris_df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    iris_df.columns = ['sl', 'sw', 'pl', 'pw', 'species']

    # map species, cluster labels to strings (for coloring plots)
    iris_df['species'] = map(str, iris_df.species)
    iris_df['cluster'] = map(str, kmeans.labels_)
    
    # plot true classes
    plot1 = (ggplot(iris_df, aes('pl', 'pw', color='species'))
        + geom_point(size=40) + ggtitle('true classes'))

    # plot cluster assignments
    plot2 = (ggplot(iris_df, aes('pl', 'pw', color='cluster'))
        + geom_point(size=40) + ggtitle('cluster assignments'))

    ggsave(filename='plot1.jpg', plot=plot1)
    ggsave(filename='plot2.jpg', plot=plot2)

    # cluster validation
    k_range = range(1, 11)
    sse, sil = list(), list()
    for k in k_range:
        kmeans_k = KMeans(n_clusters=k).fit(X)

        # store sse result
        sse.append((k, kmeans_k.inertia_))

        # store sil coeff result
        try:
            sil_coeff = silhouette_score(X, kmeans_k.labels_, metric='euclidean')
            sil.append((k, sil_coeff))

        except Exception:
            print 'warning: assigning sil coeff of 0 for k = {}'.format(k)
            sil.append((k, 0))

    # plot validation results
    sse_results = pd.DataFrame(sse, columns=['k', 'sse'])
    sil_results = pd.DataFrame(sil, columns=['k', 'sil'])

    plot3 = (ggplot(sse_results, aes('k', 'sse'))
        + geom_line()
        + scale_x_continuous(k_range)
        + xlab('num clusters')
        + ylab('inertia (total sse)')
        + ggtitle('iris cluster validation (sse)'))

    plot4 = (ggplot(sil_results, aes('k', 'sil'))
        + geom_line()
        + scale_x_continuous(k_range)
        + xlab('num clusters')
        + ylab('silhouette coeff')
        + ggtitle('iris cluster validation (sil coeff)'))

    ggsave(filename='plot3.jpg', plot=plot3)
    ggsave(filename='plot4.jpg', plot=plot4)

def get_wine():

    # unscaled wine df
    wine = pd.read_csv(WINE_FILE, sep=',')
    wine.columns = (['label']
        + ['f{}'.format(k) for k in range(1, len(wine.columns))])

    # crazy wine df -- note that clustering results are NOT scale-invariant!
    wine_crazy = wine.copy()
    wine_crazy['f10'] = wine_crazy['f10'].map(lambda k: 100000 * k)
    
    # scaled wine df
    label, data = wine.ix[:, 0], wine.ix[:, 1:]
    wine_scaled = pd.concat([pd.DataFrame(label), pd.DataFrame(preproc.scale(data))],
        axis=1)
    wine_scaled.columns = (['label']
        + ['f{}'.format(k) for k in range(1, len(wine.columns))])
    
    return wine, wine_crazy, wine_scaled

def get_clusters(df):
    X, y = df.ix[:, 1:], df.ix[:, 0]
    kmeans = KMeans(n_clusters=3, n_jobs=-1).fit(X)
    print round(silhouette_score(X, kmeans.labels_, metric='euclidean'), 3)

def wine_clusters():
    wine, wine_crazy, wine_scaled = get_wine()

    print 'silhouette coeff for wine:\t\t',
    get_clusters(wine)

    print 'silhouette coeff for wine_crazy:\t',
    get_clusters(wine_crazy)

    print 'silhouette coeff for wine_scaled:\t',
    get_clusters(wine_scaled)

if __name__ == '__main__':
    iris_clusters()
    # wine_clusters()
