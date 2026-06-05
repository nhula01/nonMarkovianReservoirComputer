#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np

def next_x(x: np.ndarray, A: np.ndarray, B: np.ndarray, u: float) -> np.ndarray:
    """
    One ESN reservoir update: x_next = tanh(Ax + Bu)
    """
    z = A @ x + B * u
    return np.tanh(z)

def esn_dynamics(inputs: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Run ESN dynamics over an input sequence.
    returns X: shape (T, N)
    """
    inputs = np.asarray(inputs, dtype=float)
    T = len(inputs)
    N = A.shape[0]
    # Random initial reservoir state in [-0.1, 0.1]
    x = 0.2 * np.random.rand(N) - 0.1
    X = np.zeros((T, N), dtype=float)
    for t in range(T):
        x = next_x(x, A, B, inputs[t])
        X[t, :] = x
    return X

def make_esn_weights(N: int,input_scale: float = 1.0,spectral_radius: float = 1.0, connectivity: float = 1.0,):
    """
    Create ESN recurrent weights A and input weights B.
    """
    # Random recurrent weights in [-1, 1]
    A = 2.0 * np.random.rand(N, N) - 1.0
    # Optional sparsity mask
    if connectivity < 1.0:
        mask = np.random.rand(N, N) < connectivity
        A *= mask
    # Rescale A to desired spectral radius
    vals = np.linalg.eigvals(A)
    rho = np.max(np.abs(vals))
    if rho > 0:
        A *= spectral_radius / rho
    # Input weights
    B = input_scale * (2.0 * np.random.rand(N) - 1.0)
    print("A norm", np.linalg.norm(A))
    print("B norm", np.linalg.norm(B))
    print("A/B ratio", np.linalg.norm(A) / np.linalg.norm(B))

    return A, B