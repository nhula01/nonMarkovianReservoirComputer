import numpy as np
from sklearn.model_selection import KFold
from joblib import load 

def get_ytrain_test_from_fold(path, n_features=36, num_classes=10):
    """
    Load fold data and return:
    y_train, y_test
    If y_train/y_test are repeated once per feature, this keeps only
    one label per sample using [::n_features].
    """
    fold_data = load(path)
    y_train = fold_data["y_train"][::n_features]
    y_test  = fold_data["y_test"][::n_features]

    y_train_10 = np.eye(num_classes)[y_train.astype(int)]
    y_test_10  = np.eye(num_classes)[y_test.astype(int)]
    return y_train, y_test, y_train_10, y_test_10

def get_Xtrain_test_from_fold(path):
    """
    Load fold data and return:
    X_mean_train, X_mean_test, X_std_train, X_std_test, 
    If y_train/y_test are repeated once per feature, this keeps only
    one label per sample using [::n_features].
    """
    fold_data = load(path)
    X_mean_train = fold_data["X_mean_train"]
    X_mean_test  = fold_data["X_mean_test"]

    X_std_train = fold_data["X_std_train"]
    X_std_test  = fold_data["X_std_test"]
    return X_mean_train, X_mean_test, X_std_train, X_std_test
    
def extract_pq(x):
    # Convert string like "(array([0.1]), array([0.2]))"
    # into two floats: P, Q
    s = str(x)
    s = s.replace("array", "np.array")
    P, Q = eval(s, {"np": np})
    return float(np.asarray(P).squeeze()), float(np.asarray(Q).squeeze())

import ast

def extract_delayed_single_column(df, column_name, N=10, repeat_steps=5):
    """
    Extract delayed P,Q features from a single column when the same input
    is repeated every `repeat_steps` simulation steps.
    Returns one feature row per repeated input block.
    For repeat_steps = 5, it returns rows:
        X[4], X[9], X[14], ...
    """
    if N % 2 != 0:
        raise ValueError("N must be even because each delayed time gives two values: P and Q.")
    T = len(df)
    num_delays = N // 2 
    pq_values = np.zeros((T, 2), dtype=float) # store all p and q values at every step
    for t in range(T): # extract p and q for each step
        P, Q = extract_pq(df[column_name].iloc[t])
        pq_values[t, 0] = P
        pq_values[t, 1] = Q 
    X = np.zeros((T, N), dtype=float) # create a feature matrix
    for t in range(T):
        delayed_pairs = []
        for d in range(num_delays):
            idx = t - d * repeat_steps
            if idx >= 0:
                delayed_pairs.append(pq_values[idx])
        delayed_pairs = delayed_pairs[::-1]
        flattened = np.array(delayed_pairs).reshape(-1)
        X[t, -len(flattened):] = flattened
    # Keep only one row per repeated-input block
    n_blocks = T // repeat_steps
    X_block = X[repeat_steps - 1 : n_blocks * repeat_steps : repeat_steps]
    return X_block

def extract_block_single_column_skip(df,column_name="P0",N_fts=36,n_samples=500,skip_steps=1):
    """
    Extract N_fts values from one column for each sample.
    Parameters
    ----------
    df : pandas DataFrame
        Data containing the reservoir outputs.
    n_samples : 500
        Number of samples to extract. 
    skip_steps : int
        Step size between extracted features inside one sample.
        skip_steps = 1:
            rows 0, 1, 2, ..., 35
        skip_steps = 5:
            rows 0, 5, 10, ..., 175
    sample_gap : int or None
        Distance between the start of consecutive samples.
        If None, defaults to N_fts * skip_steps.
    Returns
    -------
    X : np.ndarray
        Shape: (n_samples, N_fts)
    """
    df = df[column_name]
    X = np.zeros((n_samples, N_fts), dtype=float)
    idx = 0 
    for _ in range(int(n_samples)):
        for i in range(N_fts):
            idx = _*N_fts*skip_steps + i*skip_steps + (skip_steps - 1)
            value = df.iloc[idx]
            parsed = ast.literal_eval(str(value))
            X[_, i] = np.asarray(parsed[0]).squeeze().real
    return X

def extract_block_single_column_pq(df,column_name,N_fts=36,n_samples=500, skip_steps=1,mode="both"):
    """
    Extract P/Q features from one column where each row contains (P, Q).
    For each sample, collect N_fts points.
    With skip_steps=5, sample 0 uses rows:
        4, 9, 14, 19, ...
    because we collect the point at the end of each skip block.
    mode:
        "P"    -> return P only, shape (n_samples, N_fts)
        "Q"    -> return Q only, shape (n_samples, N_fts)
        "both" -> return P and Q, shape (n_samples, 2*N_fts)
    """

    df_col = df[column_name]
    if mode in ["P", "Q"]:
        X = np.zeros((n_samples, N_fts), dtype=float)
    elif mode == "both":
        X = np.zeros((n_samples, 2 * N_fts), dtype=float)
    else:
        raise ValueError("mode must be 'P', 'Q', or 'both'.")
    for sample in range(int(n_samples)):
        for i in range(N_fts):
            idx = sample * N_fts * skip_steps + i * skip_steps + (skip_steps - 1)
            P, Q = extract_pq(df_col.iloc[idx])
            if mode == "P":
                X[sample, i] = P
            elif mode == "Q":
                X[sample, i] = Q
            elif mode == "both":
                X[sample, 2*i] = P
                X[sample, 2*i + 1] = Q
    return X

def add_polynomial_features(X, nodes=10):
    """
    Add polynomial/product features to X.
    Starting from each row:
        x0, x1, ..., x9
    We append:
        x1 * x_i        for i = 0,...,nodes-1
        poly_0 * x_i    for i = 0,...,nodes-1
        poly_1 * x_last for i = 0,...,nodes-1
        poly_2 * x_last for i = 0,...,nodes-1
    """
    X_poly = []
    for row in X:
        state1 = list(row)
        # original input features
        statedf1 = list(row)
        # 1st polynomial block
        for i in range(nodes):
            state1.append(state1[1] * statedf1[i])
        # 2nd polynomial block
        for i in range(nodes):
            state1.append(state1[nodes] * statedf1[i])
        # 3rd polynomial block
        for i in range(nodes):
            state1.append(state1[nodes+1] * statedf1[-1])
        # 4th polynomial block
        for i in range(nodes):
            state1.append(state1[nodes+2] * statedf1[-1])
        X_poly.append(state1)
    return np.array(X_poly, dtype=float)