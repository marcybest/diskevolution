import numpy as np
from scipy import interpolate
from scipy import optimize
from scipy.integrate import quad,nquad
import random
from scipy.integrate import odeint
from scipy.stats import rayleigh
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import time

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

rho_p = 1

#Dead zones
alpha_tau=0.5
R_out_dz = 10000 * AU_to_cm #Originally 20
c_in_dz = 1
c_out_dz = 1
H_g_in_dz = 0.135 * AU_to_cm
H_g_out_dz = 1 * AU_to_cm


#Gap functions
def delta( q = 1e-3 , alpha = 0.001 , Mach = 20. ):
    try:
        qNL = 1.04/Mach**3
        qW = 34*qNL*(alpha*Mach)**0.5

        delt_true = (qNL/q)**0.5
        delt_false = [1.0]*len(q)

        delt = np.where(q>qNL,delt_true,delt_false)

        delt += (q/qW)**3.
        return delt
    except:
        qNL = 1.04/Mach**3
        qW = 34*qNL*(alpha*Mach)**0.5
        delt = 1.0
        if( q>qNL ):
            delt = (qNL/q)**0.5
        delt += (q/qW)**3.
        return delt

def q_tilde( q = 1e-3 , alpha = 0.001 , Mach = 20. , r = 1.0 ):
    D = 7.*Mach**1.5/alpha**0.25
    qt = q/( 1 + D**3*( r**(1./6.) - 1. )**6 )**(1./3.)
    return qt

def S_gap( q = 1e-3 , alpha = 0.001 , Mach = 20. ):
    d = delta( q , alpha , Mach )
    S = 1./( 1. + (0.45/3./3.14159)*q**2*Mach**5/alpha*d )
    return S

def one_planet( r = 1.0 , rp = 1.0 , q = 1e-3 , alpha = 0.001 , Mach = 20. ):
    x = r/rp
    qt = q_tilde( q , alpha , Mach , x )
    Sigma = S_gap( qt , alpha , Mach )
    return Sigma



#Laplace coefficients
def b_hahn(x,m,s,alpha,H):
    return np.cos(m*x)*pow(1+alpha**2-2*alpha*np.cos(x) + (1+alpha**2)*H**2,-s)

def b_soft(m,s,alpha,H):
    return (2/np.pi)*quad(b_hahn, 0, np.pi, args = (m,s,alpha,H))[0]

def phi_11(alpha,H):
    #print((1/8)*alpha*(b_soft(1,3/2,alpha,H)-3*alpha*H**2*(2+H**2)*b_soft(0,5/2,alpha,H)))
    return (1/8)*alpha*(b_soft(1,3/2,alpha,H)-3*alpha*H**2*(2+H**2)*b_soft(0,5/2,alpha,H))



def phi(alpha,H):
    return (1/8)*alpha*(b_soft(1,3/2,alpha,H))



def laplace_old(m,s,alpha,h):
    n_theta = 100 + 1
    theta = np.linspace(0,np.pi,n_theta)
    dtheta = theta[1] - theta[0]

    try:
        n = len(alpha)
    except:
        n = 1

    theta = np.tile(theta,n)
    alpha = np.reshape(np.transpose(np.tile(alpha,(n_theta,1))),n*n_theta)
    h = np.reshape(np.transpose(np.tile(h,(n_theta,1))),n*n_theta)

    dint = np.multiply(np.cos(m*theta), (1 + alpha**2 - 2*alpha*np.cos(theta) + (1+alpha**2)*h**2 )**(-s)  )
    dint = np.reshape(dint,(n,n_theta))

    #plt.plot(np.reshape(theta,101) ,np.reshape(dint,101) )
    #plt.show()


    w = np.concatenate((np.asarray([1]),np.tile(np.asarray([4,2]),int( (n_theta - 1)/2 - 1)),np.asarray([4,1])))

    dint = np.multiply(w,dint)

    return 2/np.pi * np.sum(dint,1) / 3 * dtheta


def laplace(m,s,alpha,h):
    n_theta = 1000 + 1
    theta = np.logspace(np.log10(1e-10),np.log10(np.pi),n_theta)
    ltheta = np.log(theta)
    dltheta = ltheta[1] - ltheta[0]

    try:
        n = len(alpha)
    except:
        n = 1

    theta = np.tile(theta,n)
    alpha = np.reshape(np.transpose(np.tile(alpha,(n_theta,1))),n*n_theta)

    dint = np.multiply(np.cos(m*theta), (1 + alpha**2 - 2*alpha*np.cos(theta) + (1+alpha**2)*h**2 )**(-s)  )
    dint = np.multiply(dint,dltheta*theta)
    dint = np.reshape(dint,(n,n_theta))

    #plt.plot(np.reshape(theta,101) ,np.reshape(dint,101) )
    #plt.show()


    w = np.concatenate((np.asarray([1]),np.tile(np.asarray([4,2]),int( (n_theta - 1)/2 - 1)),np.asarray([4,1])))

    dint = np.multiply(w,dint)

    return 2/np.pi * np.sum(dint,1) / 3



#ODE coefficients
def c_(r):
    return 0

def b_x(r):
    return -3/2/r**(5/2)

def a_xx(r):
    return 3/r**(5/2)


class disk:
    def __init__(self, r_in, r_out, r_in_box, r_out_box, M_star, ratio, gamma, nr, alpha_in, alpha_out, r_alpha):
        self.r_in = r_in
        self.r_out = r_out
        self.r_in_box = r_in_box
        self.r_out_box = r_out_box
        self.M_star = M_star
        self.ratio = ratio
        self.gamma = gamma

        self.nr = nr
        self.r = np.logspace(np.log10(r_in_box),np.log10(r_out_box),nr)
        self.dr = np.log(self.r[1]) - np.log(self.r[0])

        #Initial density
        self.dens0 = self.powerlaw()

        self.alpha_back = alpha_in
        self.alpha_dz = alpha_out
        self.r_alpha = r_alpha

        #For planets
        self.M_j = []
        self.a_j = []
        self.e_j = []
        self.w_j = []


    #Initial Powerlaw
    def powerlaw(self):

        r = self.r
        r_in = self.r_in
        r_out = self.r_out
        r_in_box = self.r_in_box
        r_out_box = self.r_out_box
        M_star = self.M_star
        ratio = self.ratio
        gamma = self.gamma

	    #Integral is not defined for this value
        if gamma == -2 :
            return 0
        sigma_0 = (2+gamma) * ratio * M_star / (2*np.pi*r_out**2)
        return sigma_0 * (r/r_out)**gamma * np.e**(-(r/r_out)**(2+gamma))


    #Disk properties
    def temp(self,r):
	    return 280*(r/AU_to_cm)**-0.5

    def c_s(self,r):
	    c_s2 =  5/3 * k_B * self.temp(r) / 2.3 / m_p
	    c_s = np.sqrt(c_s2)
	    return c_s

    def omega_k(self,r):
        M_star = self.M_star
        omega_k = np.sqrt(G*self.M_star/r**3)
        return omega_k

    def H_g(self,r):
        return self.c_s(r)/self.omega_k(r)

    def h(self,r):
        return self.c_s(r)/self.omega_k(r)/r

    def nu(self,r):
        #return alpha(r)*c_s(r)*0.05*r
	    return self.alpha(r)*self.c_s(r)**2/self.omega_k(r)

    def alpha(self,r):
        R_in_dz = self.r_alpha
        return  (self.alpha_dz - self.alpha_back) * ( 0.5 * (1 + np.tanh((r-R_in_dz)/(c_in_dz*H_g_in_dz))) + 0.5 * (1 + np.tanh((R_out_dz-r)/(c_out_dz*H_g_out_dz)))  ) + 2*self.alpha_back - self.alpha_dz

    def v_rel(self,e_1,e_2,w_1,w_2,r):
        v_k = r * self.omega_k(r)
        return v_k * np.sqrt ( (e_1**2 + e_2**2 - 2*e_1*e_2*np.cos(np.pi*w_1/180-np.pi*w_2/180)) )


    def plot_alpha(self):
        y  = self.alpha(self.r)
        plt.loglog(self.r/AU_to_cm,y)
        plt.xlabel('Distance (au)')
        plt.ylabel('alpha')
        plt.show()


    #Crank-Nicholson Matrices
    def CN_Matrices(self,dt):
        r = self.r
        dr = self.dr
        nr = self.nr


        # D matrix #Present
        D = []
        for i in range(len(r)):

            if i == 0: #Inner boundary condition
                A = np.zeros(len(r))
            elif i == len(r)-1:
                A = np.zeros(len(r))
            else:
                a = a_xx(r[i])
                b = b_x(r[i])
                c = c_(r[i])

                w_p = self.nu(r[i+1]) * r[i+1]**(1/2)
                w_0 = self.nu(r[i]) * r[i]**(1/2)
                w_m = self.nu(r[i-1]) * r[i-1]**(1/2)

                A = np.concatenate((np.zeros(i-1),np.array([2*a*dt*w_m - b*dt*dr*w_m, 4*dr**2 - 4*dt*a*w_0 ,2*dt*a*w_p + b*dt*dr*w_p]),np.zeros(nr-i-2)))
            D = np.append(D,A)

        D = np.matrix(np.reshape(D,(nr,nr)))


        # B matrix #Future
        B = []
        for i in range(len(r)):

            if i == 0: #Inner boundary condition
                #A = np.concatenate((np.array([1,-3,3,-1]),np.zeros(len(r)-4)))
                A = np.concatenate((np.array([1]),np.zeros(len(r)-1)))
            elif i == len(r)-1:
                #A = np.concatenate((np.zeros(len(r)-4),np.array([-1,3,-3,1])))
                A = np.concatenate((np.zeros(len(r)-1),np.array([1])))
            else:
                a = a_xx(r[i])
                b = b_x(r[i])
                c = c_(r[i])

                w_p = self.nu(r[i+1]) * r[i+1]**(1/2)
                w_0 = self.nu(r[i]) * r[i]**(1/2)
                w_m = self.nu(r[i-1]) * r[i-1]**(1/2)

                A = np.concatenate((np.zeros(i-1),np.array([-2*dt*a*w_m + dr*dt*b*w_m, 4*dr**2 + 4*a*dt*w_0, -2*a*dt*w_p - b*dt*dr*w_p]),np.zeros(nr-i-2)))
            B = np.append(B,A)

        B = np.matrix(np.reshape(B,(nr,nr)))
        B = np.linalg.inv(B)
        return B, D

    def evolve_disk(self, t_max, timestep = 1e3*year):

        nr = self.nr
        t=0
        dt = 1*year
        next_plot = 0
        delta_max=0
        y = self.dens0
        y = np.matrix(np.reshape(y,(nr,1)))

        B, D = self.CN_Matrices(dt)

        t_array=[]
        dens_array=[]

        while(t<t_max):

            if t >= next_plot:
                t_array.append(t)
                dens_array.append(y)
                next_plot+=timestep

                #print(t/1000/year, dt/1000/year)

            t+=dt
            y_aux = np.matmul(B,np.matmul(D,y))
            delta_max = max(abs((y[1:-1]-y_aux[1:-1])/y[1:-1]))
            if delta_max < 0.1:
                if dt<1000*year:
                    dt*=1.1
                    B, D = self.CN_Matrices(dt)
            #elif delta_max>0.1:
            #    if dt>1*year:
            #        dt*=1/1.1
            #        B, D = self.CN_Matrices(dt)
            y=y_aux

        self.dens = interpolate.interp2d(self.r/AU_to_cm, np.asarray(t_array)/year, dens_array, kind='cubic')
        return (self.dens)

    def plot_dens(self, nr, nt, save=False, name=''):

        r = np.logspace(np.log10(self.r_in_box),np.log10(self.r_out_box),nr)
        t = np.logspace(3,7,nt)

        colors = cm.gnuplot(np.linspace(0, 1, len(t)+1))

        i=0
        for ti in t:
            plt.loglog(r/AU_to_cm,self.dens(r/AU_to_cm,ti), c=colors[i])
            i+=1

        plt.xlim(self.r_in_box/AU_to_cm/2,self.r_out_box/AU_to_cm*2)
        plt.ylim(1e-3,1e6)

        if save:
            plt.savefig(name+'.png')
        else:
            plt.show()
        plt.clf()


    ### SELF GRAVITY ###
    def add_body(self, M_j, a_j, e_j, w_j):
        self.M_j.append(M_j)
        self.a_j.append(a_j)
        self.e_j.append(e_j)
        self.w_j.append(w_j)

    def modify_body(self,index , M_j, a_j, e_j, w_j):
        self.M_j[index] = M_j
        self.a_j[index] = a_j
        self.e_j[index] = e_j
        self.w_j[index] = w_j

    def mu_d(self,a,t):
        dens = self.dens
        a_AU = a/AU_to_cm

        aux = 2*np.pi*a*dens(a)#,t)

        #for M_j_i, a_j_i in zip(self.M_j,self.a_j):
        #    aux *= one_planet(a,a_j_i,M_j_i/self.M_star,1e-3,1/self.H)
        return aux

    def A_d_1(self,a,t,H):
        return self.mu_d(a,t)*phi_11(a/self.a_p,H)

    def A_d_2(self,a,t,H):
        return self.mu_d(a,t)*(self.a_p/a)*phi_11(self.a_p/a,H)

    def calculate_ad(self,nr,nt,t_max=1e7):
        a_array = np.logspace(-1,np.log10(5),nr)
        t_array = np.logspace(3,np.log10(t_max),nt)

        Ad_array = []

        for t in t_array: #time in years

            r = np.logspace(np.log10(self.r_in_box/AU_to_cm),np.log10(self.r_out_box/AU_to_cm),1000)
            y = self.dens(r,t)
            for a_j_i, M_j_i in zip(self.a_j,self.M_j):
                y *= one_planet(r*AU_to_cm,a_j_i,M_j_i/self.M_star,self.alpha_dz,1/self.h(r*AU_to_cm))
            dens = interpolate.interp1d(r*AU_to_cm, y, kind='cubic')

            Ad_aux = []
            print(self.compute_ad(dens,3))
            for a in a_array:
                A_d = self.compute_ad(dens,a)
                #print(a,t,A_d)
                Ad_aux.append(A_d)
            Ad_array.append(Ad_aux)

        self.ad = interpolate.interp2d(a_array, t_array, Ad_array, kind='cubic')
        return(self.ad)


    def calculate_ad_t(self,a,t): #AU, yrs
        M_j = self.M_j
        a_in = self.r_in_box
        a_out= self.r_out_box

        a_p = a*AU_to_cm
        self.a_p = a_p
        n_p=np.sqrt(G*self.M_star/a_p**3)
        self.H = self.H_g(a_p)/a_p
        A_d = 2*G/(n_p*a_p**3)*(quad(self.A_d_1,a_in,a_p,args=(t,self.H))[0]+quad(self.A_d_2,a_p,a_out,args=(t,self.H))[0])

        print(2*G/(n_p*a_p**3)*(quad(self.A_d_1,a_in,a_p,args=(t,self.H))[0]) , 2*G/(n_p*a_p**3)*(quad(self.A_d_2,a_p,a_out,args=(t,self.H))[0]))

        return(A_d)


    def compute_ad_linear(self,dens,a):

        a_p = a*AU_to_cm
        n_p=np.sqrt(G*self.M_star/a_p**3)

        r_in = self.r_in_box
        r_out = self.r_out_box

        n_AD = 100+1
        r  = np.linspace(r_in,r_out,n_AD)
        dr = r[1] - r[0]

        alpha = np.where(r>a_p,a_p/r,r/a_p)
        h = self.h(r)
        alpha_= np.where(r>a_p,a_p/r,1)

        phi = 1/8 * alpha * ( laplace(1,3/2,alpha,h) - 3*alpha*h**2 * (2+h**2)*laplace(0,5/2,alpha,h))

        dint = np.multiply( (2*np.pi*r*dr), dens(r))
        dint = np.multiply(dint,phi)
        dint = np.multiply(dint,alpha_)

        #plt.plot(r/AU_to_cm,dint)
        #plt.xscale('log')
        #plt.show()

        w = np.concatenate((np.asarray([1]),np.tile(np.asarray([4,2]),int( (n_AD - 1)/2 - 1)),np.asarray([4,1])))
        dint = np.multiply(w,dint)

        return 2*G/(n_p*a_p**3)*np.sum(dint) / 3


    def compute_ad(self,dens,a):

        a_p = a*AU_to_cm
        n_p=np.sqrt(G*self.M_star/a_p**3)

        r_in = self.r_in_box #3*AU_to_cm#
        r_out = self.r_out_box

        n_AD = 1000+1

        r_int_in  = np.logspace(np.log10(a_p),np.log10(r_in),n_AD)
        r_int_out = np.logspace(np.log10(a_p),np.log10(r_out),n_AD)

        lr_int_in = np.log(r_int_in)
        lr_int_out = np.log(r_int_out)

        dlr_in  = lr_int_in[0] - lr_int_in[1]
        dlr_out = lr_int_out[1] - lr_int_out[0]


        #Inner Integral
        alpha = r_int_in/a_p
        h =  self.h(a_p)

        phi = 1/8 * alpha * ( laplace(1,3/2,alpha,h) - 3*alpha*h**2 * (2+h**2)*laplace(0,5/2,alpha,h))

        dint = np.multiply( (2*np.pi*r_int_in**2), dens(r_int_in) )
        dint = np.multiply(dint,phi)

        dint_ = dint

        w = np.concatenate((np.asarray([1]),np.tile(np.asarray([4,2]),int( (n_AD - 1)/2 - 1)),np.asarray([4,1])))
        dint_in = np.multiply(w,dint)

        #Outer Integral

        alpha = a_p/r_int_out
        phi = 1/8 * alpha * ( laplace(1,3/2,alpha,h) - 3*alpha*h**2 * (2+h**2)*laplace(0,5/2,alpha,h))
        dint = np.multiply( (2*np.pi*r_int_out**2), dens(r_int_out) )
        dint = np.multiply(dint,phi)
        dint = np.multiply(dint,alpha)

        w = np.concatenate((np.asarray([1]),np.tile(np.asarray([4,2]),int( (n_AD - 1)/2 - 1)),np.asarray([4,1])))
        dint_out = np.multiply(w,dint)

        return 2*G/(n_p*a_p**3) /3 * ( dlr_in * np.sum(dint_in) + dlr_out * np.sum(dint_out) )


    def plot_ad(self, nr, nt, save=False, name=''):
        r = np.logspace(-1,np.log10(5),nr)
        t = np.logspace(3,7,nt)

        colors = cm.gnuplot(np.linspace(0, 1, len(t)+1))

        i=0
        for ti in t:
            plt.loglog(r,self.ad(r,ti), c=colors[i])
            i+=1

        plt.xlim(0.1,5)
        plt.yscale("symlog", linthresh=1e-11)
        plt.ylim(-1e-8,1e-8)
        if save:
            plt.savefig(name+'.png')
        else:
            plt.show()
        plt.clf()


    def save_dens(self,dens_name):
        with open(dens_name + '.pkl', 'wb') as f:
            pickle.dump(self.dens, f)

    def load_dens(self,dens_name):
        with open(dens_name + '.pkl', 'rb') as f:
            self.dens = pickle.load(f)
            return(self.dens)

    def save_ad(self,ad_name):
        with open(ad_name + '.pkl', 'wb') as f:
            pickle.dump(self.ad, f)

    def load_ad(self,ad_name):
        with open(ad_name + '.pkl', 'rb') as f:
            self.ad = pickle.load(f)
            return(self.ad)
