run mackey glass with the following:
tau = 10 # Delay
delay_steps = tau
t_renew = 1 # inject constant input for t times
t_max = 500 * t_renew # Total Time Evolution
Delta = 0 # Detuning
gamma = .1 # Decaying
omega = .15 # input strength
phi = np.pi/3 # phase
max_photon = 2 # amount of photons
max_bin = 100 # Maximal bins
delta_t = 1 # system time step
chi_max = 5

original: noperturbation -> Reservoir 1

allpertubed001. add .001 to the perturbation
allpertubed002. add .002 to the perturbation
allpertubed003. add .003 to the perturbation

perturbedat100_001 add .001 to the perturbation at 100
perturbedat100_002 add .002 to the perturbation at 100
perturbedat100_003 add .003 to the perturbation at 100

omega = .1 # input strength -> Reservoir 2

omega = .13 t_renew = 2 -> Reservoir 3 
