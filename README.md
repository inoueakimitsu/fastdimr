# fastdimr

[![Build Status](https://app.travis-ci.com/inoueakimitsu/fastdimr.svg?branch=main)](https://app.travis-ci.com/inoueakimitsu/fastdimr)
<a href="https://github.com/inoueakimitsu/fastdimr/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/inoueakimitsu/fastdimr"></a> 

`fastdimr` is a Python library designed to learn from the results of clustering and dimensionality reduction algorithms, allowing predictions for new data points in a reusable manner.

## Features

- Combine any clustering algorithm with classifiers
- Combine any dimensionality reduction algorithm with regressors
- User-friendly interface complying with scikit-learn’s API standards

## Installation

```bash
pip install fastdimr
```

## Usage

### Clustering Example

```python
from fastdimr import DistilledCluster
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
import numpy as np

# For example, using DBSCAN and MLPClassifier
dbscan = DBSCAN()
distiller = MLPClassifier(max_iter=200)
cluster = DistilledCluster(dbscan, distiller)

# Fit the data
cluster.fit(X)

# Predict clusters for new data points
estimated_clusters = cluster.predict(new_X)
```

### Dimensionality Reduction Example

```python
from fastdimr import DistilledTransformer
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

# For example, using TSNE and MultiOutputRegressor
tsne = TSNE(n_components=2)
distiller = MultiOutputRegressor(MLPRegressor(max_iter=200), n_jobs=-1)
transformer = DistilledTransformer(tsne, distiller)

# Fit the data
transformer.fit(X)

# Transform features for new data points
transformed_features = transformer.transform(new_X)
```

## Use Case: Customer Segmentation

Imagine you are working on a customer segmentation project for a retail company. The company has a dataset containing customer purchase history, demographic information, and other relevant features. The goal is to segment customers into distinct groups based on their purchase behavior and characteristics.

### Step 1: Clustering

You decide to use a clustering algorithm like DBSCAN to segment the customers. However, DBSCAN does not provide a straightforward way to predict the cluster of a new customer.

```python
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
from fastdimr import DistilledCluster

# Prepare your data
# X = ...

# Initialize the models
dbscan = DBSCAN(eps=3, min_samples=2)
distiller = MLPClassifier(max_iter=200)

# Create the DistilledCluster
cluster = DistilledCluster(dbscan, distiller)

# Fit the model
cluster.fit(X)
```

### Step 2: Predicting New Customer Segments

With `fastdimr`, you can now predict the cluster of new customers easily.

```python
# New customer data
# new_X = ...

# Predict the cluster for the new customer
new_customer_cluster = cluster.predict(new_X)
```

This use case demonstrates how `fastdimr` adds predictive capabilities to clustering algorithms, making it easier to apply the results of customer segmentation to new data.