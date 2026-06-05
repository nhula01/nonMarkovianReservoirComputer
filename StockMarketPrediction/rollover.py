import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from fitting_model import *
from ESN_model import *
from extracter import *
import numpy as np

def rollover(start, end, name, X_features, tau=1, Nfading=10):
    prices = yf.download(name,start=f"{start}-01-01",end=f"{end}-01-01")["Close"]
    dataset = prices.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset).flatten()
    M_test = end - start - 2
    results = []
    errors = []
    for test_round in range(M_test):
        train_start = start + test_round
        train_end = train_start + 2
        test_end = train_start + 3
        stock_3yr = yf.download(name,start=f"{train_start}-01-01",end=f"{test_end}-01-01")["Close"]
        stock_2yr = yf.download(name,start=f"{train_start}-01-01",end=f"{train_end}-01-01")["Close"]

        Ntotal = len(stock_3yr)
        Ntraining = len(stock_2yr) - Nfading
        Ntesting = Ntotal - Nfading - Ntraining
        K = Ntotal - tau

        # index position in the full 2014-2024 dataset
        window_start_date = stock_3yr.index[0]
        index = prices.index.get_loc(window_start_date)
        X = X_features[index : index + K, :]
        X_train = X[Nfading : Nfading + Ntraining, :]
        X_test = X[Nfading + Ntraining : Nfading + Ntraining + Ntesting, :]
        y_target = scaled_data[index + tau : index + K + tau]
        y_train = y_target[Nfading : Nfading + Ntraining]
        y_test = y_target[Nfading + Ntraining : Nfading + Ntraining + Ntesting]
        W = fitting_function(X_train, y_train)
        y_prediction = predict(X_test, W)

        error = nrmse(y_prediction, y_test)
        errors.append(error)
        results.append({
            "round": test_round,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
            "Ntotal": Ntotal,
            "Ntraining": Ntraining,
            "Ntesting": Ntesting,
            "NRMSE": error,
        })
    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)  
    print("Final result:")
    print("NRMSE values:", errors)
    print(f"Mean NRMSE: {mean_error}")
    print(f"Std NRMSE: {std_error}")
    return results, errors


def rollover_esn(start,end,name,N=50,tau=1,Nfading=100,input_scale=1.0,spectral_radius=0.95,connectivity=0.1):
    # Download full dataset first
    prices = yf.download(name,start=f"{start}-01-01",end=f"{end}-01-01")["Close"]
    dataset = prices.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset).flatten()
    # Build ESN weights
    A, B = make_esn_weights(N,input_scale=input_scale,spectral_radius=spectral_radius,connectivity=connectivity)
    # Generate ESN reservoir states for full time series
    X_features = esn_dynamics(scaled_data, A, B)
    M_test = end - start - 2
    results = []
    errors = []
    for test_round in range(M_test):
        train_start = start + test_round
        train_end = train_start + 2
        test_end = train_start + 3

        stock_3yr = yf.download(name,start=f"{train_start}-01-01",end=f"{test_end}-01-01")["Close"]
        stock_2yr = yf.download(name,start=f"{train_start}-01-01",end=f"{train_end}-01-01")["Close"]

        Ntotal = len(stock_3yr)
        Ntraining = len(stock_2yr) - Nfading
        Ntesting = Ntotal - Nfading - Ntraining
        K = Ntotal - tau

        # Index position in the full dataset
        window_start_date = stock_3yr.index[0]
        index = prices.index.get_loc(window_start_date)

        # ESN features
        X = X_features[index : index + K, :]
        X_train = X[Nfading : Nfading + Ntraining, :]
        X_test = X[Nfading + Ntraining : Nfading + Ntraining + Ntesting, :]

        # Target is shifted by tau
        y_target = scaled_data[index + tau : index + K + tau]
        y_train = y_target[Nfading : Nfading + Ntraining]
        y_test = y_target[Nfading + Ntraining : Nfading + Ntraining + Ntesting]

        # Fit readout
        W = fitting_function(X_train, y_train)
        y_prediction = predict(X_test, W)
        error = nrmse(y_prediction, y_test)
        errors.append(error)
        results.append({
            "round": test_round,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
            "Ntotal": Ntotal,
            "Ntraining": Ntraining,
            "Ntesting": Ntesting,
            "NRMSE": error,
        })

    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)
    print("Final result")
    print("NRMSE values:", errors)
    print(f"Mean NRMSE: {mean_error}")
    print(f"Std NRMSE: {std_error}")

    return results, errors

def run_esn_repeated(tasks,tasks_name,start=2014,end=2024,n_runs=100,N=30,tau=1,Nfading=10,input_scale=1.0,spectral_radius=0.95,connectivity=0.1):
    esn_performance = {}
    for task, task_label in zip(tasks, tasks_name):
        all_errors = []
        print(f"\nRunning ESN for {task_label}")
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            results, errors = rollover_esn(start=start,end=end,name=task,N=N,tau=tau,Nfading=Nfading,input_scale=input_scale,
                spectral_radius=spectral_radius,connectivity=connectivity)

            all_errors.append(errors)

        all_errors = np.array(all_errors)

        # Average each fold over 100 ESN random initializations
        mean_per_fold = np.mean(all_errors, axis=0)
        std_per_fold = np.std(all_errors, axis=0, ddof=1)

        # Final bar value: mean across folds
        mean_error = np.mean(mean_per_fold)

        # Final whisker: std across fold-averaged errors
        std_error = np.std(mean_per_fold, ddof=1)

        esn_performance[task_label] = {
            "all_errors": all_errors,
            "mean_per_fold": mean_per_fold,
            "std_per_fold": std_per_fold,
            "mean": mean_error,
            "std": std_error,
        }

        print(f"{task_label} ESN mean per fold:", mean_per_fold)
        print(f"{task_label} ESN final mean NRMSE:", mean_error)
        print(f"{task_label} ESN std across folds:", std_error)

    return esn_performance