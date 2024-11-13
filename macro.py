import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv
import matplotlib.cm as cm
from scipy import interpolate
from scipy import optimize
import mymodule as mm
import random
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

#Constants
M_sun = 1.989e33
M_earth = 5.97e27
AU_to_cm = 1.496e13
year = 3.1536e7
G = 6.67e-8
k_B = 1.38e-16
m_p = 1.67e-24
M_jup = 1.898e30 #g
year = 3.1536e7


#Disk
#Inner and outer limits of the initial gas profile
r_in = 0.1*AU_to_cm
r_out = 25*AU_to_cm
#Inner and outer limits of the box for the simulation
r_in_box = 0.1*AU_to_cm
r_out_box = 1000*AU_to_cm
M_star = 1*M_sun #Mass of the star
ratio = 0.01 #Total mass of the gaseous disk in stellar units
gamma = -1 #Initial gas profile index
nr = 1000 #Binning in semimajor axis
t_max = 5e6*year #Total run time of the simulation

#Viscosity
#Any 1D function of r can be used as alpha.
#You just need to set disk.alpha to your desired function,
#but setting the parameters here will automatically define it.
alpha_in = 1e-3
alpha_out = 1e-4
r_alpha = 3. * AU_to_cm #Transition between the two alphas

#Jupiter
M_j = 3. #Jupiter masses
a_j = 3. * AU_to_cm #Jupiter position
e_j = 0.05
w_j = 0.

file_name = 'fiducial'

disk = mm.disk(r_in, r_out, r_in_box, r_out_box, M_star, ratio, gamma, nr, alpha_in, alpha_out, r_alpha)
disk.add_body(M_j*M_jup,a_j,e_j,w_j)

print("Evolving disk...")
disk.evolve_disk(t_max)
#This function plots the interpolated function
disk.plot_dens(50,50)

#You can also save the function
#disk.save_dens(file_name+'_dens')
#And use it later
#disk.load_dens(file_name+'_dens')
