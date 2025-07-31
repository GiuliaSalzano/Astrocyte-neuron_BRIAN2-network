### Modified from M. Stimberg et al, Modeling Neuron–Glia Interactions with the Brian 2 Simulator, Springer International Publishing (2019), doi:10.1007/978-3-030-00817-8 18

import matplotlib
from brian2 import *
import pandas as pd



# MODEL PARAMETERS

## Some metrics parameters needed to establish proper connections
size = 3.75*mmeter           # Length and width of the square lattice
distance = 50*umeter         # Distance between neurons

### Neuron parameters
E_l = -60*mV                 # Leak reversal potential
g_l = 9.99*nS                # Leak conductance
E_e = 0*mV                   # Excitatory synaptic reversal potential
E_i = -80*mV                 # Inhibitory synaptic reversal potential
C_m = 198*pF                 # Membrane capacitance
tau_e = 5*ms                 # Excitatory synaptic time constant
tau_i = 10*ms                # Inhibitory synaptic time constant
tau_r = 5*ms                 # Refractory period
I_ex = 100*pA                # External current
V_th = -50*mV                # Firing threshold
V_r = E_l                    # Reset potential

### Synapse parameters
rho_c = 0.005                # Synaptic vesicle-to-extracellular space volume ratio
Y_T = 500.*mmolar            # Total vesicular neurotransmitter concentration
Omega_c = 40/second          # Neurotransmitter clearance rate
U_0__star = 0.6              # Resting synaptic release probability
Omega_f = 3.33/second        # Synaptic facilitation rate
Omega_d = 2.0/second         # Synaptic depression rate
w_e = 0.05*nS                # Excitatory synaptic conductance
w_i = 1.0*nS                 # Inhibitory synaptic conductance
# --- Presynaptic receptors
O_G = 1.5/umolar/second      # Agonist binding (activating) rate
Omega_G = 0.5/(60*second)    # Agonist release (deactivating) rate

### Astrocyte parameters
# ---  Calcium fluxes
O_P = 0.9*umolar/second      # Maximal Ca^2+ uptake rate by SERCAs
K_P = 0.05*umolar            # Ca2+ affinity of SERCAs
C_T = 2*umolar               # Total cell free Ca^2+ content
rho_A = 0.18                 # ER-to-cytoplasm volume ratio
Omega_C = 6/second           # Maximal rate of Ca^2+ release by IP_3Rs
Omega_L = 0.1/second         # Maximal rate of Ca^2+ leak from the ER
# --- IP_3R kinectics
d_1 = 0.13*umolar            # IP_3 binding affinity
d_2 = 1.05*umolar            # Ca^2+ inactivation dissociation constant
O_2 = 0.2/umolar/second      # IP_3R binding rate for Ca^2+ inhibition
d_3 = 0.9434*umolar          # IP_3 dissociation constant
d_5 = 0.08*umolar            # Ca^2+ activation dissociation constant
# --- IP_3 production
# --- Agonist-dependent IP_3 production
O_beta = 0.5*umolar/second   # Maximal rate of IP_3 production by PLCbeta
O_N = 0.3/umolar/second      # Agonist binding rate
Omega_N = 0.5/second         # Maximal inactivation rate
K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
zeta = 10                    # Maximal reduction of receptor affinity by PKC
# --- Endogenous IP3 production
O_delta = 1.2*umolar/second  # Maximal rate of IP_3 production by PLCdelta
kappa_delta = 1.5*umolar     # Inhibition constant of PLC_delta by IP_3
K_delta = 0.1*umolar         # Ca^2+ affinity of PLCdelta
# --- IP_3 degradation
Omega_5P = 0.05/second       # Maximal rate of IP_3 degradation by IP-5P
K_D = 0.7*umolar             # Ca^2+ affinity of IP3-3K
K_3K = 1.0*umolar            # IP_3 affinity of IP_3-3K
O_3K = 4.5*umolar/second     # Maximal rate of IP_3 degradation by IP_3-3K
# --- IP_3 diffusion
F = 0.09*umolar/second       # GJC IP_3 permeability
I_Theta = 0.3*umolar         # Threshold gradient for IP_3 diffusion
omega_I = 0.05*umolar        # Scaling factor of diffusion
# --- Gliotransmitter release and time course
C_Theta = 0.5*umolar         # Ca^2+ threshold for exocytosis
Omega_A = 0.6/second         # Gliotransmitter recycling rate
U_A = 0.6                    # Gliotransmitter release probability
G_T = 200*mmolar             # Total vesicular gliotransmitter concentration
rho_e = 6.5e-4               # Astrocytic vesicle-to-extracellular volume ratio
Omega_e = 60/second          # Gliotransmitter clearance rate
alpha = 0.0                  # Gliotransmission nature


# Define HF stimulus
stimulus = TimedArray([1.0, 1.0, 1.0, 1.0, 1.0], dt=10*second)

# Simulation time (based on the stimulus)
duration = 50*second          # Total simulation time

### General parameters
N_e = 3200                   # Number of excitatory neurons
N_i = 800                    # Number of inhibitory neurons
N_a = 3200                   # Number of astrocytes

### Load the adjacency matrix (needed when you do not use random connectivity)
f_read=open('./adjacency_matrix.txt', 'r')
adj=np.loadtxt(f_read)

# Model definition

### Neurons

neuron_eqs = '''
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_ex*stimulus(t))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens  # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i : siemens  # post-synaptic inhibitory conductance

# Neuron position in space
x : meter (constant)
y : meter (constant)
jit_x: meter (constant)
jit_y: meter (constant)
'''
neurons = NeuronGroup(N_e + N_i, model=neuron_eqs,
		      threshold='v>V_th', reset='v=V_r',
		      refractory='tau_r', method='euler')
exc_neurons = neurons[:N_e]
inh_neurons = neurons[N_e:]

# Arrange excitatory neurons in a grid
N_rows = int(sqrt(N_e))
N_cols = N_e/N_rows
grid_dist = (size / N_rows)
exc_neurons.jit_x = '(2*rand()-1)*grid_dist/4'
exc_neurons.jit_y = '(2*rand()-1)*grid_dist/4'
exc_neurons.x = 'int(i / N_rows)*grid_dist - N_rows/2.0*grid_dist+jit_x'
exc_neurons.y = '(i % N_rows)*grid_dist - N_cols/2.0*grid_dist+jit_y'

# Arrange inhibitory neurons in a grid
N_rows_i = int(sqrt(N_i))
f=5
inh_neurons.jit_x = '(2*rand()-1)*grid_dist/4'
inh_neurons.jit_y = '(2*rand()-1)*grid_dist/4'
inh_neurons.x = 'int(i / N_rows_i)*grid_dist/f-N_rows_i/2.0*grid_dist/f + jit_x'
inh_neurons.y = '(i % N_rows_i)*grid_dist/f- N_rows_i/2.0*grid_dist/f + jit_j'

# Random initial membrane potential values and conductances
neurons.v = 'E_l + rand()*(V_th-E_l)'
neurons.g_e = 'rand()*w_e'
neurons.g_i = 'rand()*w_i'

### Synapses

synapses_eqs = '''
# Neurotransmitter
dY_S/dt = -Omega_c * Y_S                                    : mmolar (clock-driven)
# Fraction of activated presynaptic receptors
dGamma_S/dt = O_G * G_A * (1 - Gamma_S) - Omega_G * Gamma_S : 1 (clock-driven)
# Usage of releasable neurotransmitter per single action potential:
du_S/dt = -Omega_f * u_S                                    : 1 (event-driven)
# Fraction of synaptic neurotransmitter resources available for release:
dx_S/dt = Omega_d *(1 - x_S)                                : 1 (event-driven)
U_0                                                         : 1
# released synaptic neurotransmitter resources:
r_S                                                         : 1
# gliotransmitter concentration in the extracellular space:
G_A                                                         : mmolar

# which astrocyte covers this synapse ?
x_syn : meter (constant)
y_syn : meter (constant)
'''

synapses_action = '''
U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
u_S += U_0 * (1 - u_S)
r_S = u_S * x_S
x_S -= r_S
Y_S += rho_c * Y_T * r_S
'''

exc_syn = Synapses(exc_neurons, neurons, model=synapses_eqs,
		   on_pre=synapses_action+'g_e_post += w_e*r_S',
		   method='linear')

exc_sources, exc_targets = adj[:N_e].nonzero()
exc_syn.connect(i=exc_sources, j=exc_targets)	# or exc_syn.connect(p=0.05) for random connectivity network with 5% probability of connection
exc_syn.x_S = 1.0

inh_syn = Synapses(inh_neurons, neurons, model=synapses_eqs,
		   on_pre=synapses_action+'g_i_post += w_i*r_S',
		   method='linear')
		   
inh_sources,  inh_targets = adj[N_e:].nonzero()
inh_syn.connect(i=inh_sources, j=inh_targets)	# or inh_syn.connect(p=0.2) for random connectivity network with 20% probability of connection
inh_syn.x_S = 1.0

# Connect excitatory synapses to an astrocyte depending on the position of the
# post-synaptic neuron

exc_syn.x_syn = '(x_post)'
exc_syn.y_syn = '(y_post)'

### Astrocytes
# The astrocyte emits gliotransmitter when its Ca^2+ concentration crosses a threshold

astro_eqs = '''
# Fraction of activated astrocyte receptors:
dGamma_A/dt = O_N * Y_S * (1 - clip(Gamma_A,0,1)) -
	      Omega_N*(1 + zeta * C/(C + K_KC)) * clip(Gamma_A,0,1) : 1
# Intracellular IP_3
dI/dt = J_beta + J_delta - J_3K - J_5P + J_coupling              : mmolar
J_beta = O_beta * Gamma_A                                        : mmolar/second
J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mmolar/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K)                : mmolar/second
J_5P = Omega_5P*I                                                : mmolar/second
# Diffusion between astrocytes:
J_coupling                                                       : mmolar/second

# Ca^2+-induced Ca^2+ release:
dC/dt = J_r + J_l - J_p                                   : mmolar
dh/dt = (h_inf - h)/tau_h                                 : 1
J_r = (Omega_C * m_inf**3 * h**3) * (C_T - (1 + rho_A)*C) : mmolar/second
J_l = Omega_L * (C_T - (1 + rho_A)*C)                     : mmolar/second
J_p = O_P * C**2/(C**2 + K_P**2)                          : mmolar/second
m_inf = I/(I + d_1) * C/(C + d_5)                         : 1
h_inf = Q_2/(Q_2 + C)                                     : 1
tau_h = 1/(O_2 * (Q_2 + C))                               : second
Q_2 = d_2 * (I + d_1)/(I + d_3)                           : mmolar

# Fraction of gliotransmitter resources available for release:
dx_A/dt = Omega_A * (1 - x_A) : 1
# gliotransmitter concentration in the extracellular space:
dG_A/dt = -Omega_e*G_A        : mmolar
# Neurotransmitter concentration in the extracellular space:
Y_S                           : mmolar

# The astrocyte position in space
x : meter (constant)
y : meter (constant)
jit_x: meter (constant)
jit_y: meter (constant)
'''

glio_release = '''
G_A += rho_e * G_T * U_A * x_A
x_A -= U_A *  x_A
'''
astrocytes = NeuronGroup(N_a, astro_eqs,
			 # The following formulation makes sure that gliotransmitter release is
			 # only triggered at the first threshold crossing
			 threshold='C>C_Theta',
			 refractory='C>C_Theta',
			 reset=glio_release,
			 method='rk4',
			 dt=1e-2*second)

# Arrange astrocytes in a grid
N_rows_a = int(sqrt(N_a))
N_cols_a = N_a/N_rows_a
grid_dist = size / N_rows_a
astrocytes.jit_x = '(2*rand()-1)*grid_dist/2'
astrocytes.jit_y = '(2*rand()-1)*grid_dist/2'
astrocytes.x = 'int(i / N_rows_a)*grid_dist - N_rows_a/2.0*grid_dist + jit_x' 
astrocytes.y = '(i % N_rows_a)*grid_dist - N_rows_a/2.0*grid_dist + jit_y ' 

astrocytes.C = 0.01*umolar
astrocytes.h = 0.9
astrocytes.I = 0.01*umolar
astrocytes.x_A = 1.0

astro_diam = 1.5*grid_dist	# Astocytes diameter
dist_ASyn = astro_diam*1/2	# Threshold distance between astrocytes and exc. synapses
dist_SynA = astro_diam*1/2	# Threshold distance between exc. synapses and astrocytes

# Astrocyte-to-synapse connections
ecs_astro_to_syn = Synapses(astrocytes, exc_syn,
		    'G_A_post = G_A_pre : mmolar (summed)')
ecs_astro_to_syn.connect('sqrt((x_pre-x_syn_post)**2+(y_pre-y_syn_post)**2) < dist_ASyn')

# Synapse-to-Astrocyte connections
ecs_syn_to_astro = Synapses(exc_syn, astrocytes,
		    'Y_S_post = Y_S_pre/N_incoming : mmolar (summed)')
ecs_syn_to_astro.connect('sqrt((x_syn_pre-x_post)**2+(y_syn_pre-y_post)**2) < dist_SynA', p='exp(-((x_syn_pre-x_post)**2+(y_syn_pre-y_post)**2)/(2*(dist_SynA/2)**2))')

# Diffusion between astrocytes
astro_to_astro_eqs = '''
delta_I = I_post - I_pre            : mmolar
J_coupling_post = -(1 + tanh((abs(delta_I) - I_Theta)/omega_I))*
		  sign(delta_I)*F/2 : mmolar/second (summed)
'''
astro_to_astro = Synapses(astrocytes, astrocytes,
			  model=astro_to_astro_eqs)
# Connect to all astrocytes less than astro_diam away
astro_to_astro.connect('i != j and '
		       'sqrt((x_pre-x_post)**2 +'
		       '     (y_pre-y_post)**2) < astro_diam')
		       
# MONITORS

exc_mon = SpikeMonitor(exc_neurons)
inh_mon = SpikeMonitor(inh_neurons)
ast_mon = SpikeMonitor(astrocytes)

# RUN
run(duration, report='text')
