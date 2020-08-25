import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import (
    ConvexHull,
    #  Delaunay,
    #  Voronoi,
    #  voronoi_plot_2d,
)
from sklearn.datasets import make_classification
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

#  sns.set()
#  sns.set_palette("Set2")

amount_of_data = 100
n_dimensions = 2
amount_of_samples = 6
#  sampling_method = "random"
sampling_method = "var1"

X, y = make_classification(
    n_samples=amount_of_data,
    n_features=n_dimensions,
    n_informative=n_dimensions,
    n_classes=2,
    class_sep=3,
    n_redundant=0,
    scale=0.1,
)
X = MinMaxScaler().fit_transform(X)

x_data = X[:, 0]
y_data = X[:, 1]
if n_dimensions == 3:
    z_data = X[:, 2]

fig = plt.figure()
if n_dimensions == 3:
    ax = fig.add_subplot(111, projection="3d")

if sampling_method == "random":
    sample = X[np.random.choice(len(X), size=amount_of_samples, replace=False)]
elif sampling_method == "var1":
    max_sum = 0
    # select n random points from inside and calculate distances
    for i in range(0, 100):
        random_sample = X[
            np.random.choice(len(X), size=amount_of_samples, replace=False)
        ]

        # calculate distance to each other
        total_distance = np.sum(pairwise_distances(random_sample, random_sample))
        if total_distance > max_sum:
            max_sum = total_distance
            sample = random_sample

    #  sample = hull_points
    amount_of_samples = len(sample)

random_sample = X[np.random.choice(len(X), size=amount_of_samples, replace=False)]
# variante 1: convex hull, random m points from inner, select those subset with maximum distance to each other
# variante 2: variante 1 aber ohne convex hull_points
# variante 3: var 1 oder var 2 aber inklusive 10 real random points on top
# variante 4: voronoi diagram, zufällig n der flächen samplen ->


x_sample = sample[:, 0]
y_sample = sample[:, 1]
if n_dimensions == 3:
    z_sample = sample[:, 2]

# random sample
x_random = random_sample[:, 0]
y_random = random_sample[:, 1]
if n_dimensions == 3:
    z_random = random_sample[:, 2]


x_data = np.concatenate((x_data, x_sample, x_random),)
y_data = np.concatenate((y_data, y_sample, y_random),)
if n_dimensions == 3:
    z_data = np.concatenate((z_data, z_sample, z_random),)
y = np.concatenate((y, [2] * amount_of_samples, [3] * amount_of_samples))
size = np.concatenate(
    ([10] * amount_of_data, [100] * amount_of_samples, [100] * amount_of_samples)
)

if n_dimensions == 3:
    ax.scatter(x_data, y_data, z_data, c=y, s=size)
else:
    plt.scatter(x_data, y_data, c=y, s=size)

plt.show()
