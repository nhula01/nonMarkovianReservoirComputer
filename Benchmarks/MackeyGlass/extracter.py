import numpy as np

def extract_pq(x):
    # Convert string like "(array([0.1]), array([0.2]))" into two floats: P, Q
    s = str(x)
    s = s.replace("array", "np.array")
    P, Q = eval(s, {"np": np})
    return float(np.asarray(P).squeeze()), float(np.asarray(Q).squeeze())

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
    
def extract_delayed_p_or_q_single_column(df, column_name, N=10, repeat_steps=5):
    """
    Extract delayed P-only or Q only features from a single column.
    """
    T = len(df)
    # Extract P at every simulation step
    p_values = np.zeros(T, dtype=float)
    for t in range(T):
        p_values[t] = extract_pq(df[column_name].iloc[t])[0]
    # Build delayed P feature matrix at every simulation step
    X = np.zeros((T, N), dtype=float)
    for t in range(T):
        delayed_values = []
        for d in range(N):
            idx = t - d * repeat_steps
            if idx >= 0:
                delayed_values.append(p_values[idx])
        # Oldest delay on the left, newest on the right
        delayed_values = delayed_values[::-1]
        X[t, -len(delayed_values):] = delayed_values
    # Keep only the last simulation step of each repeated-input block
    selected_indices = np.arange(repeat_steps - 1, T, repeat_steps)
    X_skip = X[selected_indices]
    return X_skip


def extract_pq_row(df, column_names=None, repeat_steps=5, keep_last=True):
    """
    Extract both P and Q from columns PQ-15, PQ-14, ..., PQ0 in each selected row.
    Output feature order:
        [P(PQ-15), Q(PQ-15), P(PQ-14), Q(PQ-14), ..., P(PQ0), Q(PQ0)]
    Output shape:
        (number_of_selected_rows, 2 * number_of_columns)
    """

    if column_names is None:
        column_names = [f"PQ-{i}" for i in range(15, 0, -1)] + ["PQ0"]

    T = len(df)

    if keep_last:
        selected_indices = np.arange(repeat_steps - 1, T, repeat_steps)
    else:
        selected_indices = np.arange(0, T, repeat_steps)

    X = np.zeros((len(selected_indices), 2 * len(column_names)), dtype=float)

    for i, row_idx in enumerate(selected_indices):
        features = []

        for col in column_names:
            P, Q = extract_pq(df[col].iloc[row_idx])
            features.extend([P, Q])

        X[i, :] = features

    return X


def extract_p_row(df, column_names, repeat_steps=5, keep_last=True):
    """
    Extract P-only features from multiple columns in the same row.
    Instead of taking delayed values from one column over time,
    this takes P values from several columns in each selected row.
    Example:
        column_names = ["P-9", "P-8", ..., "P0"]
    If repeat_steps = 5 and keep_last=True, it keeps rows:
        4, 9, 14, 19, ...
    Output shape:
        (T // repeat_steps, len(column_names))
    """
    T = len(df)
    if keep_last:
        selected_indices = np.arange(repeat_steps - 1, T, repeat_steps)
    else:
        selected_indices = np.arange(0, T, repeat_steps)
    X = np.zeros((len(selected_indices), len(column_names)), dtype=float)
    for i, row_idx in enumerate(selected_indices):
        for j, col in enumerate(column_names):
            X[i, j] = extract_pq(df[col].iloc[row_idx])[0]
    return X

def add_polynomial_features(X, nodes=10):
    """
    Add polynomial/product features to X.
    """
    X_poly = []
    for row in X:
        state1 = list(row)
        # original input features
        statedf1 = list(row)
        # 1st polynomial block
        for i in range(nodes):
            state1.append(state1[0] * statedf1[i])
        # 2nd polynomial block
        for i in range(nodes):
            state1.append(state1[nodes] * statedf1[i])
        # 3rd polynomial block
        for i in range(nodes):
            state1.append(state1[nodes*2] * statedf1[i])
        # 4th polynomial block
        for i in range(nodes):
            state1.append(state1[nodes*3] * statedf1[i])
        X_poly.append(state1)
    return np.array(X_poly, dtype=float)