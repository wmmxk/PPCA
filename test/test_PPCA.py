"""
run this test in ./test/test_PPCA.py

the test load the IRIS data and then run PPCA and visualized the points.

"""

import matplotlib.pyplot as plt
from sklearn import datasets

from lib.PPCA import PPCA


def test_PPCA():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    ppca = PPCA(latent_dim=2, max_iter=50)

    data_std = ppca.fit(X)
    data_reduced = ppca.transform_data(data_std)
    print(data_reduced.shape)

    colors = ["red", "blue", "yellow"]
    color_each_point = [colors[i] for i in y]
    print(color_each_point)
    plt.scatter(data_reduced[:, 1], data_reduced[:, 0], color=color_each_point)
    plt.show()

