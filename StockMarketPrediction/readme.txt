Parameters: 
tau=15
maxbin=300
delta_t=1
gamma = .1
epsilon = .15
theta = np.pi/3


We extract the data for Apple, NASDAQ, S&P500 for the period 01-01-2014 to 01-01-2024 which has 2516 data points.

AAPL_Features_2014_2024_rep10.csv: runing AAPL for ten years with repetition 10
SP500_Features_2014_2024_rep10.csv: runing SP500 for ten years with repetition 10
NASDAQ_Features_2014_2024_rep10.csv: runing NASDAQ for ten years with repetition 10
Similarly, this was done for 1 repetition.


After this, vary the delay tau=5, 10, 15, 20 for SP500.
Since 15 was calculated above, compute 3 more cases:
- SP500_Features_2014_2024_rep10_tau5.csv
- SP500_Features_2014_2024_rep10_tau10.csv
- SP500_Features_2014_2024_rep10_tau20.csv
Similarly, this was done for 1 repetition.
