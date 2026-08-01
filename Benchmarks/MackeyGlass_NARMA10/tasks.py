import numpy as np

def mackey_glass(Npoints=1000, delay: int = 10,tau: int = 17, x_0: float = 1.0, beta: float = 0.2,
    gamma: float = 0.1, n: int = 10, dt: float = 1.0):
    x_set = [x_0]

    # 1000 warm-up steps to let the dynamics settle
    for i in range(1, Npoints + 1000 + 1):
        if i <= tau:
            xpast = 0
        else:
            # Python index is 0-based, so subtract 1
            xpast = x_set[i - tau - 1]
            
        x_0 += dt * (beta * xpast / (1 + xpast**n) - gamma * x_0)
        x_set.append(x_0)
    return x_set[1001 - delay:]

import numpy as np

def narma(Npoints=1000,order=10,delay=10,seed=0,):
    """
    Generate a NARMA time series.
    -------
    y : ndarray
        NARMA target sequence.
    u : ndarray
        Input sequence used to generate y.
    """
    rng = np.random.default_rng(seed)
    # Random input in [0, 0.5]
    u = rng.uniform(0.0, 0.5, Npoints + 1000 + order + 1)
    y = np.zeros_like(u)
    # Warm-up + generation
    for t in range(order, len(u)-1):
        y[t+1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t-order+1:t+1])
            + 1.5 * u[t-order+1] * u[t]
            + 0.1)
    # Remove warm-up
    start = 1000 + delay
    end = start + Npoints

    return y[start:end], u[start:end]
    
def sine_square_input_task(Nsegments = 110, wsin = 10.0, Nsin = 8, seed= 1234):
    rng = np.random.default_rng(seed)
    # input timing
    T = 2 * np.pi / wsin   # one period
    dt = T / Nsin          # time step to fit Nsin points in one period
    # total periods
    NT = Nsegments
    tstart = 0.0
    tend = NT * T - dt
    tlist = np.linspace(tstart, tend, NT * Nsin)
    # one binary label per period
    y_binary = rng.integers(0, 2, size=NT)
    # repeat each label Nsin times
    y_bar = np.repeat(y_binary, Nsin)
    # build input sequence
    u = np.zeros(len(tlist), dtype=float)
    for n in range(len(tlist)):
        t = tlist[n]
        if y_bar[n] == 1:
            # sine input
            u[n] = np.sin(wsin * t) + 1.0
        elif y_bar[n] == 0:
            # square input: first half high, second half low
            n_in_period = n % Nsin
            if n_in_period < Nsin // 2:
                u[n] = 2.0
            else:
                u[n] = 0.0
        elif y_bar[n] == -1:
            # constant input
            u[n] = 1.0
        elif y_bar[n] == 2:
            # smooth positive sine
            u[n] = 0.5 * (np.sin(wsin * t) + 1.0)
    return u, y_bar