#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
def fitting_function(X: np.ndarray, y: np.ndarray, delta: float = 1e-10) -> np.ndarray:
    """
    Ridge regression with bias column.
    X: shape (L, nnodes)
    y: shape (L,)
    returns W: shape (nnodes + 1,)
    """
    L, nnodes = X.shape
    # Add bias column
    Xb = np.hstack([np.ones((L, 1)), X])
    # Ridge matrix
    A = Xb.T @ Xb + delta * np.eye(nnodes + 1)
    # Solve A W = Xb.T y
    W = np.linalg.solve(A, Xb.T @ y)
    return W

def predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Predict using learned weights W.
    """
    L = X.shape[0]
    Xb = np.hstack([np.ones((L, 1)), X])
    return Xb @ W

def nrmse(ypred: np.ndarray, ytarget: np.ndarray) -> float:
    """
    Normalized root mean squared error.
    """
    denom = np.max(ytarget) - np.min(ytarget)
    return np.sqrt(np.mean((ypred - ytarget) ** 2)) / denom

