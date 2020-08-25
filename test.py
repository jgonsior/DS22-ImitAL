import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial import (
    ConvexHull,
    convex_hull_plot_2d,
    #  Delaunay,
    #  Voronoi,
    #  voronoi_plot_2d,
)
import seaborn as sns

#  sns.set()
#  sns.set_palette("Set2")

amount_of_data = 1000
n_dimensions = 2
amount_of_samples = 20
sampling_method = "random"

X, y = make_classification(
    n_samples=amount_of_data,
    n_features=n_dimensions,
    n_informative=n_dimensions,
    n_classes=2,
    #  class_sep=2,
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

# variante 1: convex hull, random m points from inner, select those subset with maximum distance to each other
# variante 2: variante 1 aber ohne convex hull_points
# variante 3: var 1 oder var 2 aber inklusive 10 real random points on top
# variante 4: voronoi diagram, zufällig n der flächen samplen ->

hull = ConvexHull(X)
print(hull.simplices)


# select n points from hull


#  exit(-1)

#  sample = MinMaxScaler().fit_transform(sample)
#
#  # find those samples who are closest to the generated sample
#  for s in sample:
#      print(pairwise_distances(X, np.reshape(s, (-1,1)))

#  exit(-2)

x_sample = sample[:, 0]
y_sample = sample[:, 1]
if n_dimensions == 3:
    z_sample = sample[:, 2]


x_data = np.concatenate((x_data, x_sample),)
y_data = np.concatenate((y_data, y_sample),)

# random sample


if n_dimensions == 3:
    z_data = np.concatenate((z_data, z_sample),)

y = np.concatenate((y, [2] * amount_of_samples))
size = np.concatenate(([10] * amount_of_data, [100] * amount_of_samples))
print(size)

if n_dimensions == 3:
    ax.scatter(x_data, y_data, z_data, c=y)
else:
    plt.scatter(x_data, y_data, c=y, s=size)
#
for simplex in hull.simplices:
    plt.plot(X[simplex, 0], X[simplex, 1], "k-")

#  plt.scatter(x_data, y_data, c=y)

plt.show()
