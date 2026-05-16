tau = 10 # Delay
delay_steps = tau
Delta = 0 # Detuning
gamma = .1 # Decaying
epsilon = .15 # input strength
max_photon = 2 # amount of photons
max_bin = 100 # Maximal bins
delta_t = 1 # system time step

# constant drive
MeasurementsBond5_gammatau10.csv: chi_max = 5, tau=100, phi=0
MeasurementsBond10_gammatau10.csv: chi_max = 10, tau=100, phi=0
MeasurementsBond15_gammatau10.csv: chi_max = 15, tau=100, phi=0

MeasurementsBond5_gammatau10.csv: chi_max = 5, tau=50, phi=0
MeasurementsBond10_gammatau10.csv: chi_max = 10, tau=50, phi=0

MeasurementsBond5_gammatau1.csv: chi_max = 5, tau=10, phi=0
MeasurementsBond10_gammatau1.csv: chi_max = 10, tau=10, phi=0
MeasurementsBond15_gammatau1.csv: chi_max = 5, tau=10, phi=0


# random drive
MeasurementsBond5_gammatau1_5_random.csv: chi_max = 5, tau=15, phi=0 random.seed = 43
MeasurementsBond10_gammatau1_5_random.csv: chi_max = 10, tau=15, phi=0 random.seed = 43

# random drive with each input repeated 5 times
MeasurementsBond5_gammatau1_5_random_masked.csv: chi_max = 5, tau=15, phi=0 random.seed = 43
MeasurementsBond10_gammatau1_5_random_masked.csv: chi_max = 10, tau=15, phi=0 random.seed = 43

# Different phase phi
MeasurementsBond10_differentquadratures_masked.csv: chi_max=10, tau=15, phi=np.pi/3 random input repeated 5 times
MeasurementsBond10_differentquadratures.csv: chi_max=10, tau=15, phi=np.pi/3