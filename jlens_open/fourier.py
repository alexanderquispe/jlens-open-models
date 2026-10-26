"""Fourier analysis over Z/n (irreps of the cyclic group) for features like days and months."""
import numpy as np


def pair_energies(x):
    """Energy per frequency pair k = 1..n//2 of mean-centered rows x (n, d) (irreps of Z/n)."""
    n = len(x)
    coeffs = np.fft.fft(x - x.mean(0), axis=0) / n
    e = (np.abs(coeffs) ** 2).sum(1)
    return np.array([e[k] + (e[n - k] if k != n - k else 0) for k in range(1, n // 2 + 1)])


def circle_plane(centroids):
    """Orthonormal basis (d, 2) of the k = 1 Fourier plane of the centroids (n, d)."""
    c1 = np.fft.fft(centroids - centroids.mean(0), axis=0)[1]
    return np.linalg.qr(np.stack([c1.real, c1.imag], 1))[0]

