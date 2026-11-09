"""Probabilistic scoring rules for 3-way (H/D/A) forecasts.

All functions take `p` with shape (n, 3) in H, D, A order and integer labels `y`
(H=0, D=1, A=2). Lower is better for every score except accuracy.
"""
import numpy as np


def log_loss(p, y):
    """Mean negative log-probability given to what actually happened."""
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))))


def brier(p, y):
    """Multi-class Brier score: squared error between the probability vector and the one-hot outcome (0-2 scale)."""
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def rps(p, y):
    """Ranked probability score. Respects the order H > D > A: calling a draw when
    the home side won is a smaller error than calling an away win."""
    onehot = np.eye(3)[y]
    cum_p, cum_o = np.cumsum(p, axis=1)[:, :2], np.cumsum(onehot, axis=1)[:, :2]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def accuracy(p, y):
    return float(np.mean(p.argmax(axis=1) == y))


def per_match_log_loss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))


def paired_bootstrap_diff(loss_a, loss_b, n_boot=2000, seed=0):
    """95% interval of mean(loss_a - loss_b), resampling matches with replacement."""
    rng = np.random.default_rng(seed)
    d = loss_a - loss_b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
