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
r_in = 0.1*AU_to_cm
r_out = 25*AU_to_cm
r_in_box = 0.1*AU_to_cm
r_out_box = 1000*AU_to_cm
M_star = 1.07*M_sun #Mstar should be 1.249
ratio=0.01 #Total mass of the disk in stellar units
gamma=-1
nr=1000
t_max = 5e6*year

#Viscosity - inital alpha_in=1e-3 and alpha_out=1e-4
alpha_in = 1e-3
alpha_out = 1e-4
r_alpha = 3.23 * AU_to_cm #Preset is 3,always set to be the same as a_j

#Jupiter - all presets are 3 here except eccentricity which is 0.05
M_j = 1.34 #Jupiter masses 0.68
a_j = 3.23 * AU_to_cm #0.85
e_j = 0.01 #0.28
w_j=0.

#Planetesimal distribution -- Usually 2 here
alpha_p = 3



file_name = 'fiducial'#'30_' + str(int(round(alpha_in*1e6)))+'_'+str(int(round(alpha_out*1e6)))+'_'+str(int(round(M_j*100)))

disk = mm.disk(r_in, r_out, r_in_box, r_out_box, M_star, ratio, gamma, nr, alpha_in, alpha_out, r_alpha)
disk.add_body(M_j*M_jup,a_j,e_j,w_j)



'''
#Comment out once run
print("Evolving disk...")
disk.evolve_disk(t_max)

disk.save_dens(file_name+'_dens')

print("Calculating precession...")
disk.calculate_ad(30,30,t_max = t_max/year)

disk.save_ad(file_name+'_ad')
'''

disk.load_dens(file_name+'_dens')
disk.load_ad(file_name+'_ad')


#Solids -- Was 0.1 - 1, 3 values 5,7 rho usually 1
rho_p = 1
r_p = np.logspace(5,7,10) #in cm
a_p = np.logspace(np.log10(0.1*AU_to_cm),np.log10(1.8*AU_to_cm),30)


A = np.pi*(a_p[1:]**2 - a_p[:-1]**2)
r_mid = (a_p[1:] + a_p[:-1])/2
y = disk.dens(r_mid/AU_to_cm,0) #position in AU, and time in years
M = np.multiply(y, A) / 100
pla_array=[]

for a_p_i, m_p_i in zip(a_p,M):
    for r_p_i in r_p:
        pla_array.append(disk.planetesimal(r_p_i, rho_p, a_p_i, m_p=m_p_i/len(r_p)))

disk.evolve_planetesimal(pla_array,t_max, 5000)


with open(file_name+'_pla.npy', 'wb') as f:
    np.save(f, pla_array)


with open(file_name+'_pla.npy', 'rb') as f:
    pla_array = np.load(f,allow_pickle=True)

disk.plot_ae(pla_array,'Kepler-139-AE-N-1e3-1e4-Trial')
#disk.plot_dens(pla_array,save=True,name='KOI-85-Density-N')
#disk.plot_ad(pla_array,save=True,name='KOI-85-AD-N')
#disk.plot_aw(pla_array,'KOI-85-AW-N')
#plt.show()
