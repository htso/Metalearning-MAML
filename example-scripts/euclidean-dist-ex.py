import matplotlib
import numpy
import perfplot
from scipy.spatial import distance


def linalg_norm(data):
    a, b = data
    return numpy.linalg.norm(a-b, axis=1)


def sqrt_sum(data):
    a, b = data
    return numpy.sqrt(numpy.sum((a-b)**2, axis=1))


def scipy_distance(data):
    a, b = data
    return list(map(distance.euclidean, a, b))


def mpl_dist(data):
    a, b = data
    return list(map(matplotlib.mlab.dist, a, b))


def sqrt_einsum(data):
    a, b = data
    a_min_b = a - b
    return numpy.sqrt(numpy.einsum('ij,ij->i', a_min_b, a_min_b))


perfplot.show(
    setup=lambda n: numpy.random.rand(2, n, 3),
    n_range=[2**k for k in range(20)],
    kernels=[linalg_norm, scipy_distance, mpl_dist, sqrt_sum, sqrt_einsum],
    logx=True,
    logy=True,
    xlabel='len(x), len(y)'
    )