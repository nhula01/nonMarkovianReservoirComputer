tau = 10 # Delay
delay_steps = tau
t_renew = 3 # inject constant input for t times
t_max = 1000 * t_renew # Total Time Evolution
Delta = 0 # Detuning
gamma = .1 # Decaying
omega = .15 # input strength
phi = np.pi/3 # phase
max_photon = 2 # amount of photons
max_bin = 15 # Maximal bins
delta_t = 1 # system time step
chi_max = 5

we change omega as [.08, .1,.12,.14, .16]
we do multiple reps [1,3,5,8,10] 