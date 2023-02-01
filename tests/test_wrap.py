import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.datasets import make_regression

from fastdimr import DistilledTransformer, DistilledCluster


def test_distilled_transformer():

    X1, y1 = make_regression(n_samples=1000, n_features=4, n_informative=10, n_targets=1, noise=0.1, random_state=123)
    X2, y2 = make_regression(n_samples=1000, n_features=4, n_informative=10, n_targets=1, noise=0.1, random_state=456)
    X = np.concatenate([X1, X2 + 1])
    y = np.concatenate([y1, y2])

    tsne = TSNE(n_components=2)
    distiller = MultiOutputRegressor(MLPRegressor(max_iter=200), n_jobs=-1)

    transformer = DistilledTransformer(tsne, distiller)
    transformer.fit(X)
    estimated = transformer.transform(X)
    print(estimated)
    estimated = transformer.transform(X)
    print(estimated)  


def test_distilled_cluster():

    X1, y1 = make_regression(n_samples=1000, n_features=4, n_informative=10, n_targets=1, noise=0.1, random_state=123)
    X2, y2 = make_regression(n_samples=1000, n_features=4, n_informative=10, n_targets=1, noise=0.1, random_state=456)
    X = np.concatenate([X1, X2 + 1])
    y = np.concatenate([y1, y2])

    dbscan = DBSCAN()
    distiller = MLPClassifier(max_iter=200)
    cluster = DistilledCluster(dbscan, distiller)    
    cluster.fit(X)
    estimated = cluster.fit_transform(X)
    print(estimated)
    estimated = cluster.transform(X)
    print(estimated)

    print(pd.crosstab(DBSCAN().fit_predict(X), cluster.fit_predict(X)))

if __name__ == '__main__':
    test_distilled_cluster()
    test_distilled_transformer()
