parameters: 
tau=15
maxbin=300
delta_t=1
gamma = .1
epsilon = .15
theta = np.pi/3




01-01-2020 to 01-01-2025
BTC_features_10rep_phipithird.csv: t_renew=10, phi=np.pi/3
BTC_features_1rep_phi0.csv: t_renew=1, phi=0
BTC_features_5rep_phi0.csv: t_renew=5, phi=0
BTC_features_5rep_phipithird.csv: t_renew=5, phi=np.pi/3
Similarly for SP500. The code for SP500: ^GPSC and for BTC: BTC-USD on yf.

----------------------------------------
01-01-2014 to 01-01-2024
I will run the following

AAPL_Features_2014_2024_rep10.py: runing AAPL for ten years and perform rollover calculation for the error bar with rep 10
- AAPL_Features_2014_2024_rep10.csv
SP500_Features_2014_2024_rep10.py: runing SP500 for ten years and perform rollover calculation for the error bar with rep 10
- SP500_Features_2014_2024_rep10.csv
BTC_Features_2014_2024_rep10.py: runing BTC for ten years and perform rollover calculation for the error bar with rep 10
- BTC_Features_2014_2024_rep10.csv
NASDAQ_Features_2014_2024_rep10.py: runing NASDAQ for ten years and perform rollover calculation for the error bar with rep 10
- NASDAQ_Features_2014_2024_rep10.csv
After this, I will vary the delay tau=5, 10, 15, 20, 25 for SP500
Since 15 was calculated above, we calculate 3 more cases:
SP500_Features_2014_2024_rep10_tau5.py:
- SP500_Features_2014_2024_rep10_tau5.csv
SP500_Features_2014_2024_rep10_tau10.py:
- SP500_Features_2014_2024_rep10_tau10.csv
SP500_Features_2014_2024_rep10_tau20.py:
- SP500_Features_2014_2024_rep10_tau20.csv
SP500_Features_2014_2024_rep10_tau25.py:
- SP500_Features_2014_2024_rep10_tau25.csv


I run the following with 1 rep as well
