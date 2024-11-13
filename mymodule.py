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

    def total_mass(self, t):
        r = self.r

        r_in = r[r<self.a_j]
        r_out = r[r>self.a_j]

        #M_in
        r_mid = (r_in[1:] + r_in[:-1])/2
        A = np.pi*(r_in[1:]**2 - r_in[:-1]**2)

        H = self.h(r_mid)
        sigma = self.dens(r_mid/AU_to_cm,t) * one_planet(r_mid,self.a_j[0],self.M_j[0]/self.M_star,self.alpha_dz,1/H)

        M_in = np.sum(A*sigma)

        #M_out
        r_mid = (r_out[1:] + r_out[:-1])/2
        A = np.pi*(r_out[1:]**2 - r_out[:-1]**2)

        H = self.h(r_mid)
        sigma = self.dens(r_mid/AU_to_cm,t) * one_planet(r_mid,self.a_j[0],self.M_j[0]/self.M_star,self.alpha_dz,1/H)

        M_out = np.sum(A*sigma)

        return M_in, M_out

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



    ##### PLANETESIMALS ######

    class planetesimal:
        def __init__(self, r_p, rho_p, a_p_0, m_p):
                self.r_p = r_p
                self.rho_p = rho_p
                self.a_p_0 = a_p_0
                r_H = 2 * (2*(4*np.pi/3 * (r_p**3) * rho_p )/(3*M_sun))**(1/3)
                self.e_p_0 = rayleigh.rvs(size=1, scale = r_H)
                self.w_p_0 = random.uniform(0,2*np.pi)
                self.m_p = m_p

    def evolve_planetesimal(self, planetesimals, t_max, nt):

        t_max = t_max/year

        t = np.logspace(3,np.log10(t_max),nt)

        for planetesimal in planetesimals:
            print(planetesimal.a_p_0/AU_to_cm,planetesimal.r_p)
            h_0=planetesimal.e_p_0*np.cos(planetesimal.w_p_0-self.w_j[0])
            k_0=planetesimal.e_p_0*np.sin(planetesimal.w_p_0-self.w_j[0])

            result = odeint( self.planetesimal_ode, y0=[planetesimal.a_p_0,h_0,k_0], t=t, args=(planetesimal.r_p,planetesimal.r_p) )
            result = result.transpose()
            a = result[0]/AU_to_cm
            e = np.sqrt(result[1]**2 + result[2]**2)
            w = 180/np.pi * np.arctan2(result[1],result[2])

            planetesimal.a = a
            planetesimal.e = e
            planetesimal.w = w
            planetesimal.t = t

            #plt.plot(t,a)


    def planetesimal_ode(self, params, t, r, r_0):

        dens = self.dens
        Ad = self.ad

        #TODO eliminate this
        M_c = self.M_star

        a=params[0]
        h=params[1]
        k=params[2]

        #e=params[1]
        #w=params[2]%(2*np.pi)
        #w_j=params[3]%(2*np.pi)


        #Calculate Reynolds number correctly
        '''
        nu_m = self.c_s(a)/2e-15/1e12
        Re = 2*r_0*a*self.omega_k(a) / nu_m
        if Re > 800:
            C_D = 0.44
        elif Re < 1:
            C_D = 24/Re
        else:
            C_D = 24*Re**(-0.6)
        '''

        C_D = 0.44

        delta_r = 0.001 * a/AU_to_cm #AU

        #rho = self.dens(a/AU_to_cm,t)/2/self.H_g(a)
        #rho_1 = self.dens(a/AU_to_cm + delta_r,t)/2/self.H_g(a + delta_r*AU_to_cm)

        rho = one_planet(a,self.a_j[0],self.M_j[0]/self.M_star,self.alpha_dz,a/self.H_g(a))*self.dens(a/AU_to_cm,t)/2/self.H_g(a)
        rho_1 = one_planet(a+delta_r*AU_to_cm,self.a_j[0],self.M_j[0]/self.M_star,self.alpha_dz,a/self.H_g(a+delta_r*AU_to_cm))*self.dens(a/AU_to_cm + delta_r,t)/2/self.H_g(a + delta_r*AU_to_cm)

        drho = ( rho_1 - rho ) / delta_r

        alpha = - drho/rho * a/AU_to_cm
        beta = 3/7

        v_k = np.sqrt(G*M_c/a)
        tau_0 = 8/3/C_D * rho_p/rho * r_0/v_k
        tau_0 *= 1/year
        eta = (alpha + beta)/2*(self.H_g(a)/a)**2

        n=np.sqrt(G*M_c/a**3)

        A_pp =  (n/4)*(self.M_j[0]/M_c)*(a/self.a_j[0])**2*b_soft(1,1.5,a/self.a_j[0],0) * year
        A_pj = -(n/4)*(self.M_j[0]/M_c)*(a/self.a_j[0])**2*b_soft(2,1.5,a/self.a_j[0],0) * year
        A_dp = Ad(a/AU_to_cm,t)[0] * year
        A_dj = Ad(self.a_j[0]/AU_to_cm,t)[0] * year

        #da = -2*a/tau_0*np.sqrt(5/8*e**2 + eta**2)*(eta + (alpha/4+5/16) * e**2)
        #de = A_pj * e_j * np.sin(w - w_j) - e/tau_0*np.sqrt(5/8*e**2 + eta**2)
        #dw = A_dp + A_pp + A_pj * e_j/e * np.cos(w - w_j)
        #dwj= A_dj

        plt.scatter(a/AU_to_cm,a/tau_0)

        da = -2*a/tau_0*np.sqrt(5/8*(h**2 + k**2) + eta**2)*(eta + (alpha/4+5/16) * (h**2+k**2))
        de_dt = A_pj*self.e_j[0]*h/np.sqrt(h**2 + k**2) - np.sqrt(h**2 + k**2)/tau_0 * np.sqrt(5/8*(h**2+k**2)+eta**2)
        dw_dt = A_pp + A_pj*self.e_j[0]*k/(h**2 + k**2) + A_dp - A_dj

        dh = de_dt*h/np.sqrt(h**2 + k**2) + k*dw_dt
        dk = de_dt*k/np.sqrt(h**2 + k**2) - h*dw_dt

        #if abs(dw-dwj) < 1e-4 :
        #    print(t,dw-dwj,A_d+A_pp,dwj,A_pj * e_j/e * np.cos(w - w_j))
        #print( - 180/np.pi * np.arctan(  (-eta/tau_0)/A_pp    ))
        #print(t,a/da)
        #print(A_pp + Ad(a,t_aux)[0] - Ad(a_j/AU_to_cm,t_aux)[0])

        #return [da,de,dw,dwj]
        return [da[0],dh,dk]

    def analytical(self,a,t_p,r):

        a_p = a*AU_to_cm

        #Calculate Reynolds number correctly
        C_D=0.44

        delta_r = 0.001 * a #AU

        #TODO: Add gap
        rho =  self.dens(a,t_p)/2/self.H_g(a_p)
        rho_1 = self.dens(a + delta_r,t_p)/2/self.H_g(a_p + delta_r*AU_to_cm)

        drho = ( rho_1 - rho ) / delta_r
        alpha = - drho/rho * a
        beta = 3/7

        v_k = np.sqrt(G*1*M_sun/a_p)
        tau_0 = 8/3/C_D * rho_p/rho * r/v_k
        tau_0 *= 1/year
        eta = (alpha + beta)/2*(self.H_g(a_p)/a_p)**2

        A_pp = []
        A_pj = []

        for a_pi in a_p:
            n=np.sqrt(G*1*M_sun/a_pi**3)
            A_pp.append((n/4)*(self.M_j[0]/M_sun)*(a_pi/self.a_j[0])**2*laplace(1,3/2,a_pi/self.a_j[0],0)[0] * year)
            A_pj.append(-(n/4)*(self.M_j[0]/M_sun)*(a_pi/self.a_j[0])**2*laplace(2,3/2,a_pi/self.a_j[0],0)[0] * year)

        A_pp = np.asarray(A_pp)
        A_pj = np.asarray(A_pj)

        A = A_pj*self.e_j[0]
        B = A_pp + self.ad(a,t_p)*year - self.ad(self.a_j[0]/AU_to_cm,t_p)[0]*year

        a_eq = 5/8/tau_0**2
        b_eq = B**2 + eta**2/tau_0**2
        c_eq = -A**2

        e2 = (-b_eq + np.sqrt(b_eq**2-4*a_eq*c_eq))/2/a_eq
        e = np.sqrt(e2)

        w = 180/np.pi*np.arctan2(1/tau_0/A * np.sqrt( 5/8*e2 + eta**2),-B/A)
        w[w<0] += 360.
        #A_pj* self.e_j[0]/e * np.cos(w*np.pi/180.)

        #print(self.ad(a,t_p)*year, self.ad(self.a_j[0]/AU_to_cm,t_p)[0]*year )

        return e, w


    def plot_ae(self, planetesimals, name, nt=50):

        r = np.logspace(np.log10(0.1),np.log10(5),500)
        r_sol = np.logspace(np.log10(0.11),np.log10(2.5),500)
        os.system('mkdir tmp')
        t = planetesimals[0].t

        order = np.arange(0,len(t),len(t)/nt)
        order = np.concatenate((order,np.asarray([len(t)-1])))
        order = order.astype(int)

        for i in order:

            print(i)

            fig, axs = plt.subplots(1,2,sharex=True,constrained_layout=True)
            t = planetesimals[0].t
            a=[]
            e=[]
            w=[]
            r_0=[]
            m=[]
            t=t[i]

            for planetesimal in planetesimals:
                a.append(planetesimal.a[i])
                e.append(planetesimal.e[i])
                angle = planetesimal.w[i]
                if angle<0:
                    angle+=360
                w.append(angle)
                r_0.append(np.log10(planetesimal.r_p/1e4)**3)
                m.append(planetesimal.m_p)

            r_hist = np.logspace(np.log10(0.1),np.log10(1.5),25)
            r_mid = (r_hist[1:] + r_hist[:-1]) / 2
            hist,bins = np.histogram(a,bins=r_hist,weights=m)
            area = np.pi*(r_hist[1:]**2 - r_hist[:-1]**2) * AU_to_cm**2

            sol_e,sol_w = self.analytical(r_sol,t,1e5)
            axs[0].plot(r_sol,sol_e,'r--',alpha=0.4)
            #axs[1,0].plot(r_sol,sol_w,'r--',alpha=0.4)

            sol_e,sol_w = self.analytical(r_sol,t,1e6)
            axs[0].plot(r_sol,sol_e,'g--',alpha=0.4)
            #axs[1,0].plot(r_sol,sol_w,'g--',alpha=0.4)

            sol_e,sol_w = self.analytical(r_sol,t,1e7)
            axs[0].plot(r_sol,sol_e,'b--',alpha=0.4)
            #axs[1,0].plot(r_sol,sol_w,'b--',alpha=0.4)


            #ae plot

            print('There are ' ,len(a))

            axs[0].scatter(a,e,c='k',s=r_0)
            time = "%.2f" % (t/1e6)
            axs[0].text(0.13,1e-1,time + ' Myr',size=20, weight='medium')
            axs[0].set_box_aspect(1)
            axs[0].set_xlim(0.05,5)
            axs[0].set_ylim(1e-5,1e0)
            axs[0].set_xscale('log')
            axs[0].set_yscale('log')
            axs[0].set_ylabel('Eccentricity')

            '''
            #aw plot
            axs[1,0].scatter(a,w,c='k',s=r_0)
            axs[1,0].text(0.15,1e-2,time + ' Myr',size=20, weight='medium')
            axs[1,0].set_box_aspect(1)
            axs[1,0].set_xlim(0.05,5)
            axs[1,0].set_ylim(0,360)
            axs[1,0].set_xscale('log')
            axs[1,0].set_ylabel('Omega')
            '''

            b=[]
            for ri in r:
                ri = ri*AU_to_cm
                b.append(one_planet(ri,self.a_j[0],self.M_j[0]/self.M_star,self.alpha_dz,ri/self.H_g(ri)))

            axs[1].plot(r,self.dens(r,t),'k--')
            axs[1].fill_between(r,self.dens(r,t),np.multiply(self.dens(r,t),b) ,color='tab:red')
            axs[1].fill_between(r,np.multiply(self.dens(r,t),b),np.zeros(len(r)),color='tab:blue')
            axs[1].axhline(y=1686028,xmin=0.05,xmax=0.0985)
            axs[1].axhline(y=115056,xmin=0.0985,xmax=0.3565)
            axs[1].axhline(y=4241,xmin=0.3565,xmax=2)
            
            axs[1].plot(r_mid,hist/area,'k')

            axs[1].set_box_aspect(1)
            axs[1].set_xscale('log')
            axs[1].set_yscale('log')
            axs[1].set_xlim(0.05,5)
            axs[1].set_ylim(1e-3,1e4)
            axs[1].set_ylabel('Surface density (g/cm2)')
            axs[1].set_xlabel('Semimajor axis (au)')

            fig.set_size_inches(10, 5)
            #fig.suptitle('Ratio: '+ str(self.ratio) + ' M_j: ' + str(self.M_j[0]/M_jup) + ' M_star: ' + str(self.M_star/M_sun) + ' alpha_in: ' + str(self.alpha_back) + ' alpha_out: ' + str(self.alpha_dz) )
            plt.savefig('tmp/'+ '%05d' % i +'_'+str(0)+'.png',dpi=100)
            plt.cla()
            plt.close()

        os.chdir('tmp')
        os.system('convert -delay 10 -loop 0 ./*.png ../'+name+'.gif')
        os.chdir('..')
        os.system('rm -r tmp')



    def plot_aw(self, planetesimals, name, nt=50):

        r = np.logspace(np.log10(0.1),np.log10(5),500)
        r_sol = np.logspace(np.log10(0.11),np.log10(2.5),500)
        os.system('mkdir tmp')
        t = planetesimals[0].t

        order = np.arange(0,len(t),len(t)/nt)
        order = np.concatenate((order,np.asarray([len(t)-1])))
        order = order.astype(int)

        for i in order:

            print(i)

            fig, axs = plt.subplots(1,2,sharex=True,constrained_layout=True)
            t = planetesimals[0].t
            a=[]
            e=[]
            w=[]
            r_0=[]
            m=[]
            t=t[i]

            for planetesimal in planetesimals:
                a.append(planetesimal.a[i])
                e.append(planetesimal.e[i])
                angle = planetesimal.w[i]
                if angle<0:
                    angle+=360
                w.append(angle)
                r_0.append(np.log10(planetesimal.r_p/1e4)**3)
                m.append(planetesimal.n_p*4*np.pi/3*(planetesimal.r_p/2)**3 * planetesimal.rho_p)

            r_hist = np.logspace(np.log10(0.1),np.log10(1.5),25)
            r_mid = (r_hist[1:] + r_hist[:-1]) / 2
            hist,bins = np.histogram(a,bins=r_hist,weights=m)
            area = np.pi*(r_hist[1:]**2 - r_hist[:-1]**2) * AU_to_cm**2

            sol_e,sol_w = self.analytical(r_sol,t,1e5)
            axs[0].plot(r_sol,sol_w,'r--',alpha=0.4)
            #axs[1,0].plot(r_sol,sol_w,'r--',alpha=0.4)

            sol_e,sol_w = self.analytical(r_sol,t,1e6)
            axs[0].plot(r_sol,sol_w,'g--',alpha=0.4)
            #axs[1,0].plot(r_sol,sol_w,'g--',alpha=0.4)

            sol_e,sol_w = self.analytical(r_sol,t,1e7)
            axs[0].plot(r_sol,sol_w,'b--',alpha=0.4)
            #axs[1,0].plot(r_sol,sol_w,'b--',alpha=0.4)


            #aw plot
            axs[0].scatter(a,w,c='k',s=r_0)
            time = "%.2f" % (t/1e6)
            axs[0].text(0.15,1e-2,time + ' Myr',size=20, weight='medium')
            axs[0].set_box_aspect(1)
            axs[0].set_xlim(0.05,5)
            axs[0].set_ylim(0,360)
            axs[0].set_xscale('log')
            axs[0].set_ylabel('Omega')


            b=[]
            for ri in r:
                ri = ri*AU_to_cm
                b.append(one_planet(ri,self.a_j[0],self.M_j[0]/self.M_star,self.alpha_dz,ri/self.H_g(ri)))

            axs[1].plot(r,self.dens(r,t),'k--')
            axs[1].fill_between(r,self.dens(r,t),np.multiply(self.dens(r,t),b) ,color='tab:red')
            axs[1].fill_between(r,np.multiply(self.dens(r,t),b),np.zeros(len(r)),color='tab:blue')

            axs[1].plot(r_mid,hist/area,'k')

            axs[1].set_box_aspect(1)
            axs[1].set_xscale('log')
            axs[1].set_yscale('log')
            axs[1].set_xlim(0.05,5)
            axs[1].set_ylim(1e-3,1e4)
            axs[1].set_ylabel('Surface density (g/cm2)')
            axs[1].set_xlabel('Semimajor axis (au)')

            fig.set_size_inches(10, 5)
            fig.suptitle('Ratio: '+ str(self.ratio) + ' M_j: ' + str(self.M_j[0]/M_jup) + ' M_star: ' + str(self.M_star/M_sun) + ' alpha_in: ' + str(self.alpha_back) + ' alpha_out: ' + str(self.alpha_dz) )
            plt.savefig('tmp/'+ '%05d' % i +'_'+str(0)+'.png',dpi=100)
            plt.cla()
            plt.close()

        os.chdir('tmp')
        os.system('convert -delay 10 -loop 0 ./*.png ../'+name+'.gif')
        os.chdir('..')
        os.system('rm -r tmp')




    def plot_v_rel(self, planetesimals, name, a_min,a_max, nt=50):

        a_r = (a_min+a_max)/2

        r = np.logspace(np.log10(0.1),np.log10(5),500)
        r_sol = np.logspace(np.log10(0.11),np.log10(2.5),500)
        t = planetesimals[0].t

        order = np.arange(0,len(t),len(t)/nt)
        order = np.concatenate((order,np.asarray([len(t)-1])))
        order = order.astype(int)

        Q=[]
        t_Q=[]

        for i in order:

            t = planetesimals[0].t
            a=[]
            e=[]
            r_prom=[]
            w=[]
            r_0=[]
            m=[]
            n=[]
            t=t[i]

            for planetesimal in planetesimals:
                a.append(planetesimal.a[i])
                e.append(planetesimal.e[i])
                r_prom.append(planetesimal.a[i] * (1 + 1/2*planetesimal.e[i]**2))
                angle = planetesimal.w[i]
                if angle<0:
                    angle+=360
                w.append(angle)
                r_0.append(np.log10(planetesimal.r_p/1e4)**3)
                n.append(planetesimal.n_p)
                m.append(planetesimal.n_p*4*np.pi/3*(planetesimal.r_p/2)**3 * planetesimal.rho_p)

            r_prom = np.asarray(r_prom)
            a = np.asarray(a)
            w = np.asarray(w)
            e = np.asarray(e)
            m = np.asarray(m)
            n = np.asarray(n)

            e = e[np.where(r_prom>a_min)]
            w = w[np.where(r_prom>a_min)]
            m = m[np.where(r_prom>a_min)]
            n = n[np.where(r_prom>a_min)]
            a = a[np.where(r_prom>a_min)]
            r_prom = r_prom[np.where(r_prom>a_min)]

            e = e[np.where(r_prom<a_max)]
            w = w[np.where(r_prom<a_max)]
            m = m[np.where(r_prom<a_max)]
            n = n[np.where(r_prom<a_max)]
            a = a[np.where(r_prom<a_max)]
            r_prom = r_prom[np.where(r_prom<a_max)]

            '''
            #Method 1
            v_disp=0.
            n_total=0.
            if(len(m)>2):
                for j in range(len(m)):
                    for k in range(j+1,len(m)):
                        e_1 = e[j]
                        w_1 = w[j]

                        e_2 = e[k]
                        w_2 = w[k]

                        #print(e_1,e_2,w_1,w_2)

                        v_disp  += n[j]*n[k]*(self.v_rel(e_1,e_2,w_1,w_2,a_r*AU_to_cm))**2
                        n_total += n[j]*n[k]
                v_disp_prev = np.sqrt(v_disp)/np.sqrt(n_total)
            '''

            #Method 2
            e_x = []
            e_y = []
            for j in range(len(m)):
                e_x.append(n[j] * m[j] * e[j] * np.cos(np.pi/180*w[j]))
                e_y.append(n[j] * m[j] * e[j] * np.sin(np.pi/180*w[j]))

            m_total = np.sum(np.multiply(n,m))

            e_x_0 = np.sum(e_x)/m_total
            e_y_0 = np.sum(e_y)/m_total

            e_cm = np.sqrt(e_x_0**2 + e_y_0**2)
            w_cm = 180/np.pi*np.arctan2(e_y_0,e_x_0)

            v_disp=0.
            for j in range(len(m)):
                v_disp  += n[j]*m[j]*(self.v_rel(e[j],e_cm,w[j],w_cm,a_r*AU_to_cm))**2

            v_disp = np.sqrt(v_disp)/np.sqrt(m_total)

            #print('Result: ',v_disp_prev,v_disp)
            #print('------------------------')


            m_p=0
            for m_i in m:
                m_p+=m_i

            A_p = np.pi*(a_max**2 - a_min**2)*AU_to_cm**2

            sigma_p = m_p/A_p



            #v_disp = self.omega_k(a_r*AU_to_cm) * (a_r*AU_to_cm) * np.mean(e)

            M = 1e20
            v_k = a_r*AU_to_cm * self.omega_k(a_r*AU_to_cm)
            sigma_e = v_disp / v_k
            h = (M/3/M_sun)**(1/3)

            t_growth = 4.8e3 * (sigma_e/h)**2 * (M/1e26)**(1/3)  * (1/2)**(1/3) * (10/sigma_p) * a_r**0.5

            Q.append(t_growth)#v_disp*self.omega_k(a_r*AU_to_cm)/(np.pi*G*sigma_p))
            t_Q.append(t)

        return t_Q, Q


    def plot_v_esc(self, planetesimals, name, nt=50):

        r = np.logspace(np.log10(0.1),np.log10(5),500)
        r_sol = np.logspace(np.log10(0.11),np.log10(2.5),500)
        t = planetesimals[0].t

        order = np.arange(0,len(t),len(t)/nt)
        order = np.concatenate((order,np.asarray([len(t)-1])))
        order = order.astype(int)

        Q=[]
        t_Q=[]

        for i in order:

            t = planetesimals[0].t
            a=[]
            e=[]
            w=[]
            r=[]
            m=[]
            n=[]
            t=t[i]

            for planetesimal in planetesimals:
                a.append(planetesimal.a[i])
                e.append(planetesimal.e[i])
                angle = planetesimal.w[i]
                if angle<0:
                    angle+=360
                w.append(angle)
                r.append(planetesimal.r_p)
                n.append(planetesimal.n_p)
                m.append(4*np.pi/3*(planetesimal.r_p/2)**3 * planetesimal.rho_p)

            r_H = 0.01
            deltas = []
            for j in range(len(a)):
                for k in range(j+1,len(a)):
                    #r_H_1 = a[j] * 0.01 #* np.sqrt(m[j]/3/M_sun)
                    #r_H_2 = a[k] * 0.01 #* np.sqrt(m[k]/3/M_sun)
                    #r_H = max(r_H_1,r_H_2)
                    #if abs(a[j] - a[k]) < r_H:
                    #    print('-----------------')
                    #    print(a[j],a[k],e[j],e[k],w[j],w[k])

                    #    print(v_rel,v_esc)
                    if (a[j]<0.201 and a[j]>0.199 and a[k]<0.201 and a[k]>0.199 ) :
                        v_rel = self.v_rel(e[j],e[k],w[j],w[k],a[j]*AU_to_cm)
                        v_esc = np.sqrt(2*G*(m[j]+m[k])/(r[j]+r[k]))
                        delta = np.log10(v_rel/v_esc)
                        deltas.append(delta)
            print(t)
            plt.hist(deltas)
            plt.show()



    def plot_fig_2(self, planetesimals, name, nt=50):

        n_skip = 1

        r = np.logspace(np.log10(0.1),np.log10(5),100)
        r_sol = np.logspace(np.log10(0.11),np.log10(2.5),100)
        t = planetesimals[0].t


        fig, axs = plt.subplots(3,3,sharex=True,constrained_layout=True,gridspec_kw={'height_ratios': [3, 3, 1], 'width_ratios': [1, 1, 1]})

        i_array = [0,3000,4999]

        for j in [0,1,2]:
            i = i_array[j]
            a=[]
            e=[]
            w=[]
            r_0=[]
            m=[]
            time=t[i]
            print(time)


            for planetesimal in planetesimals:

                a.append(planetesimal.a[i])
                e.append(planetesimal.e[i])
                angle = planetesimal.w[i]
                w.append(np.cos(angle*np.pi/180))
                r_0.append(np.log10(planetesimal.r_p/1e4)**3)
                m.append(planetesimal.m_p)

            #collisions


            s = 1e5
            v_frag = 17.5 * 1e2 * ((s*1e-2)**-0.36 + (s*1e-5)**1.4)**(2/3)
            M_c = 1.989e33
            v_k = np.sqrt(G*M_c/(r*AU_to_cm))
            e_stir = v_frag/1.4/v_k
            axs[0,j].plot(r,e_stir,linestyle='-.',c='tab:red')

            s = 1e6
            v_frag = 17.5 * 1e2 * ((s*1e-2)**-0.36 + (s*1e-5)**1.4)**(2/3)
            M_c = 1.989e33
            v_k = np.sqrt(G*M_c/(r*AU_to_cm))
            e_stir = v_frag/1.4/v_k
            axs[0,j].plot(r,e_stir,linestyle='-.',c='tab:green')

            s = 1e7
            v_frag = 17.5 * 1e2 * ((s*1e-2)**-0.36 + (s*1e-5)**1.4)**(2/3)
            M_c = 1.989e33
            v_k = np.sqrt(G*M_c/(r*AU_to_cm))
            e_stir = v_frag/1.4/v_k
            axs[0,j].plot(r,e_stir,linestyle='-.',c='tab:blue')



            r_hist = np.logspace(np.log10(0.1),np.log10(1.5),50)
            r_mid = (r_hist[1:] + r_hist[:-1]) / 2
            hist,bins = np.histogram(a,bins=r_hist,weights=m)
            area = np.pi*(r_hist[1:]**2 - r_hist[:-1]**2) * AU_to_cm**2

            sol_e,sol_w = self.analytical(r_sol,time,1e5)
            axs[0,j].plot(r_sol,sol_e,'r--',alpha=0.4)
            axs[2,j].plot(r_sol,np.cos(sol_w*np.pi/180),'r--',alpha=0.4)

            sol_e,sol_w = self.analytical(r_sol,time,1e6)
            axs[0,j].plot(r_sol,sol_e,'g--',alpha=0.4)
            axs[2,j].plot(r_sol,np.cos(sol_w*np.pi/180),'g--',alpha=0.4)

            sol_e,sol_w = self.analytical(r_sol,time,1e7)
            axs[0,j].plot(r_sol,sol_e,'b--',alpha=0.4)
            axs[2,j].plot(r_sol,np.cos(sol_w*np.pi/180),'b--',alpha=0.4)

            #ae plot
            axs[0,j].scatter(a[::n_skip],e[::n_skip],c='k',s=r_0[::n_skip])
            time_txt = "%.2f" % (time/1e6)
            axs[0,j].text(0.13,1e-1,time_txt + ' Myr',size=20, weight='medium')
            axs[0,j].set_xlim(0.05,5)
            #axs[0,j].set_ylim(1e-5,1e0)
            axs[0,j].set_xscale('log')
            axs[0,j].set_yscale('log')
            axs[0,0].set_ylabel('Eccentricity',fontsize=20)

            #aw plot
            axs[2,j].scatter(a[::n_skip],w[::n_skip],c='k',s=r_0[::n_skip])
            axs[2,j].set_xlim(0.05,5)
            axs[2,j].set_ylim(-1,1)
            axs[2,j].set_xscale('log')
            axs[2,0].set_ylabel('cos(Δϖ)',fontsize=20)
            axs[2,j].set_xlabel('Semimajor axis (au)',fontsize=20)

            b=[]
            for ri in r:
                ri = ri*AU_to_cm
                b.append(one_planet(ri,self.a_j[0],self.M_j[0]/self.M_star,self.alpha_dz,ri/self.H_g(ri)))
            axs[1,j].plot(r,self.dens(r,time),'k--')
            axs[1,j].fill_between(r,self.dens(r,time),np.multiply(self.dens(r,time),b) ,color='tab:red')
            axs[1,j].fill_between(r,np.multiply(self.dens(r,time),b),np.zeros(len(r)),color='tab:blue')
            axs[1,j].plot(r_mid,hist/area,'k')
            axs[1,j].set_xscale('log')
            axs[1,j].set_yscale('log')
            axs[1,j].set_xlim(0.05,5)
            axs[1,j].set_ylim(1e-3,1e4)
            axs[1,0].set_ylabel('Surface density (g/cm2)',fontsize=20)


            #Invisible
            axs[0,1].get_yaxis().set_visible(False)
            axs[1,1].get_yaxis().set_visible(False)
            axs[2,1].get_yaxis().set_visible(False)
            axs[0,2].get_yaxis().set_visible(False)
            axs[1,2].get_yaxis().set_visible(False)
            axs[2,2].get_yaxis().set_visible(False)

            axs[0,0].tick_params(axis='y',labelsize=20)
            axs[1,0].tick_params(axis='y',labelsize=20)
            axs[2,0].tick_params(axis='y',labelsize=20)

            axs[2,0].tick_params(axis='x',labelsize=20)
            axs[2,1].tick_params(axis='x',labelsize=20)
            axs[2,2].tick_params(axis='x',labelsize=20)

        #fig.set_size_inches(10, 5)
        plt.show()





    def plot_fig_3(self, planetesimals, name, nt=50):


        t = planetesimals[0].t
        i_array = [2500,3000,3970,4502,4999]
        c_array = ["tab:orange","tab:red","tab:green","tab:blue","tab:purple"]

        for j in range(len(i_array)):
            i = i_array[j]
            a=[]
            e=[]
            w=[]
            r_0=[]
            m=[]
            time=t[i]


            r_p_sizes = np.logspace(5,7,10)

            for planetesimal in planetesimals:
                if True:#planetesimal.r_p < r_p_sizes[3]:
                    a.append(planetesimal.a[i])
                    e.append(planetesimal.e[i])
                    m.append((10/10)*planetesimal.m_p)


            r = np.logspace(np.log10(0.1),np.log10(2.0),50)
            y_e,y_w=disk.analytical(self,r,time,1e7)
            dy = y_e[1:] - y_e[:-1]
            asign = np.sign(dy)
            signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
            try:
                ind = np.where(signchange==1)[0]
                ind = ind[np.argmax(y_e[ind])]
                around = y_e[ind -5:(ind+1) +5]
                max_e = max(y_e[ind -5:(ind+1) +5])
                if y_e[ind] == max_e:
                    maxima = r[ind]
                else:
                    maxima = 0
            except:
                maxima = 0
                max_e = 0

            print(time,maxima)

            r_hist = np.logspace(np.log10(0.1),np.log10(1.5),50)
            r_mid = (r_hist[1:] + r_hist[:-1]) / 2
            hist,bins = np.histogram(a,bins=r_hist,weights=m)
            area = np.pi*(r_hist[1:]**2 - r_hist[:-1]**2) * AU_to_cm**2

            time_txt = "%.2f" % (time/1e6)


            plt.fill_between(r_mid,y1=hist/area,y2=0.01*self.dens(r_mid,0),alpha=0.2,color=c_array[j],label=time_txt+' Myr',linewidth=1.5)
            plt.axvline(maxima,color=c_array[j],linestyle='--')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(0.1,1.5)
            plt.ylim(1e-1,1e3)
            plt.ylabel('Surface density (g/cm2)',fontsize=15)
            plt.xlabel('Semimajor axis (au)',fontsize=15)
            plt.legend()

        #fig.set_size_inches(10, 5)
        plt.show()


    def transported_mass(self, planetesimals, name, nt=50):

        t = planetesimals[0].t


        ew_array = []
        time_array = []
        transition_array = []

        for i in range(len(t))[::100]:
            a=[]
            e=[]
            w=[]
            r_0=[]
            m=[]
            time=t[i]


            for planetesimal in planetesimals:

                a.append(planetesimal.a[i])
                e.append(planetesimal.e[i])
                m.append(planetesimal.m_p)


            r_hist = np.logspace(np.log10(0.1),np.log10(1.5),100)
            r_mid = (r_hist[1:] + r_hist[:-1]) / 2
            hist,bins = np.histogram(a,bins=r_hist,weights=m)
            area = np.pi*(r_hist[1:]**2 - r_hist[:-1]**2) * AU_to_cm**2

            y = hist - 0.01*self.dens(r_mid,0)*area
            asign = np.sign(y)
            signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
            ind = np.where(signchange==1)

            ew_max=0
            for ind_i in ind[0]:
                #Transported mass
                ew = 0
                ew_total = 0
                for k in range(len(r_mid)):
                    if (r_mid[k]>r_mid[ind_i]):
                        ew +=  0.01*self.dens(r_mid[k],0)*area[k] - hist[k]
                    ew_total += 0.01*self.dens(r_mid[k],0)*area[k]
                if ew_max < ew:
                    transition = r_mid[ind_i]
                    ew_max = ew
            ew_array.append(ew_max/M_earth/1.8)
            time_array.append(time/1e6)
            transition_array.append(transition)

        #return ew_max/M_earth

        cm = plt.cm.get_cmap('plasma')
        plt.plot(time_array,ew_array, 'k--')
        sc = plt.scatter(time_array,ew_array, c=transition_array, vmin=0.1, vmax=1.5, cmap=cm)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Time (Myr)")
        plt.ylabel("Transported Mass (Mⴲ)")
        plt.colorbar(sc)
        #plt.show()






'''
        log_step = (np.log10(t[-1]) - np.log10(t[0]))/len(t)
        trans = 1e6
        pos_trans = (np.log10(trans) - np.log10(t[0]))/log_step - 1

        new_step = int(round(2*len(t)/nt))

        order = np.arange(0,int(round(pos_trans)),new_step)
        order = np.append(order,int(round(pos_trans)))

        order = order.astype(int)

        order_log = np.exp( np.linspace( np.log(pos_trans), np.log(len(t)), int(nt/2) ) ).astype(int)

        order = np.concatenate((order,order_log))
'''


'''
                if int(planetesimal.r_p/1e5) == 1:
                    color = 'b'
                elif int(planetesimal.r_p/1e5) == 3:
                    color = 'r'
                elif int(planetesimal.r_p/1e5) == 10:
                    color = 'g'
                elif int(planetesimal.r_p/1e5) == 30:
                    color = 'm'
                elif int(planetesimal.r_p/1e5) == 100:
                    color = 'k'
                else:
                    color = 'k'
'''
