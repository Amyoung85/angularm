# -*- coding: utf-8 -*-
####################################################################################################
# Import libraries

#! /usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Colormap
import numpy as np
import scipy
import pandas as pd
import pdb
from scipy.stats import chisquare as chi2
from scipy.integrate import simps as simps
from scipy.integrate import trapz as trapz
from joblib import Parallel, delayed  
import multiprocessing
import time
time1=time.time()


#####################################################################################################
#filenames = ( 




#####################################################################################################
#  Define constants
G = 6.67408*10**(-11.)/10.**(9.)*1.98855e30 #km^3/M_sun/s^2
#G = 4.302*10**(-3.) #pc/M_sun*(km^2/s^2)
r_vir2152 = .2568
r_vir207 = .526
halo_2152_offset = 565608.
halo_207_offset = 0.573044155e6
h = 0.73
sft = 1370.*3.085677581e13 # in km
num_cores = multiprocessing.cpu_count()


######################################################################################################
#Analysis options
cutoff = 'False' # truncate data at certain radius r_vir
bulk = 'True' # to subtract bulk motion with certain percent of r_vi
bulk_percent = 0.1
coordinate_shift = 'True' #shift coordinates to center of halo
bound='False' #remove particles with E>0
showspherical = 'False' # prints spherical velocity components after conversion from cartesian
showsigma = 'False' #prints spherical component velocity dispersions
checkmass = 'False' #checks to see if mass enclosed is in correct order
radius_sort = 'False' #sorts particles in increasing radius for later analysis

#####output supression
#graph supression false will show the graph, true will hide it
supress_velocity_anisotropy = 'False'
supress_basic_histograms = 'True'
supress_EL2 = 'False'
supress_2Dhist = 'True'
supress_circle_analysis = 'False'
supress_circleEL2 = 'False'
supress_EL2_histograms = 'False'
supress_mass_enclosed = 'True'

######################################################################################################

########################### READ IN DATA #########################################33

#get position data
positions = np.genfromtxt('pos207.txt',names=['x','y','z']) #positions are Mpc/h and h=0.73
#print data


# define position components in kms
pc = 3.085677581e19 #Mpc in km 
#pc = 1
x = positions['x']*pc #in km
y = positions['y']*pc #in km
z = positions['z']*pc #in km
print 'Total number or particles', len(x)

# define center of halo to shift positions
origin = (x[0], y[0], z[0])
x0 = origin[0]
y0 = origin[1]
z0 = origin[2]

#define displacement vector for each particle
if coordinate_shift=='True':
    origin = (x[0], y[0], z[0])
    x0 = origin[0]
    y0 = origin[1]
    z0 = origin[2]
    r = np.array([x-x0,y-y0,z-z0])
else:
    x0 = 0.
    y0 = 0.
    z0 = 0.
    r = np.array([x-x0,y-y0,z-z0])
#print r

# find length of each displacement vector
r_length = np.zeros((len(x)))
for i in range(len(r.T)):
    r_length[i] = np.linalg.norm(r.T[i,:])
    
#if cutoff is true, what is maximum radius
if cutoff == 'True':
    r_vir = 0.05*pc
    #r_vir = 0.526*pc #in km
    #r_vir = 0.2568*pc #in km
else:
    r_vir=max(r_length)#should be max value    

#truncate data to include particles within r_vir if cutoff==True
if cutoff == 'True':
    #print len(x)
    cond2 = r_length<=r_vir
    x = x[cond2]
    y = y[cond2]
    z = z[cond2]
    r = np.array([x-x0,y-y0,z-z0])
    r_length = r_length[cond2]
#see how many particles you are keeping from truncation
print 'Number or particles being used', len(x)


###########get potential data
potentials = np.genfromtxt('pot207.txt') #potential are (km/s)^2, particle mass is 6.885e6 Msun/h = 9.4315 Msun
massp = 9.4315e6# particle mass is 6.885e6 Msun/h = 9.4315e6 Msun
potentials = potentials + halo_207_offset

#get potentials
if cutoff == 'True':
    phi = potentials[cond2]#*massp
else:
    phi = potentials#*massp
#print phi
maxphi = max(potentials)
minphi = min(potentials)
minphi = potentials[0]

##########get velocity data
velocities = np.genfromtxt('vel207.txt', names=['vx','vy','vz']) # in km/s

vx = velocities['vx']
vy = velocities['vy']
vz = velocities['vz']

#truncate data to include particles within r_vir if cutoff==True
if cutoff == 'True':
    #print len(vx)
    vx = vx[cond2]
    vy = vy[cond2]
    vz = vz[cond2]
    v = np.array([vx,vy,vz])
else:
    #define velocity vector
    v = np.array([vx,vy,vz])


# calculate the average velocity within the inner bulk_percent*r_vir to subtract bulk motion
if bulk == 'True':
    nvx = []
    nvy = []
    nvz = []
    for j in range(len(r_length)):
        if r_length[j]<bulk_percent*r_vir:
            #print j
            nvx = np.append(nvx,vx[j])
            nvy = np.append(nvy,vy[j])
            nvz = np.append(nvz,vz[j])
    pp = len(nvx)
    print 'Number of particles included in bulk velocity calculation', pp, 'within', bulk_percent,'of r_vir=', r_vir,'km'
    #print 'this is nvr', nvr
    #calculate average of velocity components stored
    vx_ave0 = np.average(nvx)
    vy_ave0 = np.average(nvy)
    vz_ave0 = np.average(nvz)
else: #assign all zero values if not subtracting bulk
    vx_ave0 = 0.
    vy_ave0 = 0.
    vz_ave0 = 0.
v = np.array([vx-vx_ave0,vy-vy_ave0,vz-vz_ave0])
#find magnitude of velocity for each particle
v_length = np.zeros((len(vx)))
for i in range(len(v.T)):
    v_length[i] = np.linalg.norm(v.T[i,:])



ind = np.argmax(r_length)
print ind,r.T[ind],v.T[ind],r_length[ind],np.cross(r.T[ind],v.T[ind]), np.dot(np.cross(r.T[ind],v.T[ind]),np.cross(r.T[ind],v.T[ind]))

##################################################################################################
# calculate velocity anisotropies in km

# calculate spherical velocity components

#coordinate transformation and shift to center of halo with x in km
r_km = np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)
vr = (vx-vx_ave0)*(x-x0)/r_km+(vy-vy_ave0)*(y-y0)/r_km+(vz-vz_ave0)*(z-z0)/r_km
vphi = (-(vx-vx_ave0)*(y-y0)+(vy-vy_ave0)*(x-x0))/(np.sqrt((x-x0)**2+(y-y0)**2))
vtheta = ((vx-vx_ave0)*(x-x0)*(z-z0)+(vy-vy_ave0)*(y-y0)*(z-z0))/(r_km*np.sqrt(((x-x0))**2+((y-y0))**2))-(vz-vz_ave0)*np.sqrt(((x-x0))**2+((y-y0))**2)/r_km


# assign center particle zero velocity since dividing by 0 for center particle
vr[0] = 0.
vtheta[0] = 0.
vphi[0] = 0.



# check conversion happened correctly (bulk motion subtraction needs to be turned off for this)
nv = np.array([vr,vphi,vtheta])
if bulk == 'False':
    nv_length = np.zeros((len(vr)))
    #print v,nv
    for i in range(len(nv.T)):
        nv_length[i] = np.linalg.norm(nv.T[i,:])
    v_length[0]=0
    nv_length[0]=0
    #print (v_length==nv_length).all(), v_length,nv_length,len(v_length),len(nv_length)
    print 'Conversion to spherical cooridnates preserved velocity magnitude:', np.allclose(v_length,nv_length)
if showspherical == 'True':
    print 'spherical components'
    print nv


##############calculate velocity dispersions ##################################

nbins = 40. #define number of radial bins
step = r_vir/nbins #size of each bin
#define bins across radius in pc
rbins = np.arange(step,r_vir+step,step)#in pc
#define centers of bins
rpoints = np.arange(step/2.,r_vir+step/2.,step)#in pc

#initialize
nvr = []
nvtheta = []
nvphi = []
vr_ave = np.zeros(len(rbins))
vtheta_ave = np.zeros(len(rbins))
vphi_ave = np.zeros(len(rbins))
sigma_r = np.zeros(len(rbins))
sigma_theta = np.zeros(len(rbins))
sigma_phi = np.zeros(len(rbins))

'''
#find bulk motion in spherical components by taking average of components within 10% of virial radius (would need to rewrite earlier code to make sure bulk is not subtracted twice)
for j in range(len(r_km)):
    if r_km[j]<bulk_percent*r_vir:
        #print j
        nvr = np.append(nvr,vr[j])
        nvtheta = np.append(nvtheta,vtheta[j])
        nvphi = np.append(nvphi,vphi[j])
pp = len(nvr)
print pp

#print 'this is nvr', nvr
nvr[0] = 0
nvtheta[0] = 0
nvphi[0] = 0
vr_ave0 = np.average(nvr)
vtheta_ave0 = np.average(nvtheta)
vphi_ave0 = np.average(nvphi)
     
#reinitialize    
nvr = []
nvtheta = []
nvphi = []

#print vr_ave0, vtheta_ave0, vphi_ave0
#print np.sqrt(vr_ave0**2+vtheta_ave0**2+vphi_ave0**2),np.sqrt(vx_ave0**2+vy_ave0**2+vz_ave0**2)

#'''



# find dispersion and averages within radial bins 
print 'Binning velocity dispersion in radial bins'
for i in range(1,len(rbins)):
    if i%20. == 0:
        print i,'out of',int(nbins)
    for j in range(len(r_km)):
        if r_km[j]<rbins[i]:
            if r_km[j]>rbins[i-1]:
                #print j
                nvr = np.append(nvr,vr[j])#-vr_ave0)
                nvtheta = np.append(nvtheta,vtheta[j])#-vtheta_ave0)
                nvphi = np.append(nvphi,vphi[j])#-vphi_ave0)
    sigma_r[i] = np.std(nvr)
    sigma_theta[i] = np.std(nvtheta)
    sigma_phi[i] = np.std(nvphi)

    vr_ave[i] = np.average(nvr)
    vtheta_ave[i] = np.average(nvtheta)
    vphi_ave[i] = np.average(nvphi)
    
    nvr = []
    nvtheta = []
    nvphi = []



beta = 1.-((sigma_theta**2+sigma_phi**2)/(2.*sigma_r**2))
print 'beta'
print beta
if showsigma == 'True':
    print 'velocity dispersions r, theta, phi'
    print sigma_r, sigma_theta, sigma_phi


############ analysis for orbital energies and angular momentum ###################

# sort data by increasing r after processing
if radius_sort == 'True':
    checkmass == 'True'
    dtype = [('x',float), ('y',float), ('z',float), ('r_length',float), ('vx',float), ('vy',float), ('vz',float), ('v_length',float), ('phi',float)]
    values = zip(x-x0,y-y0,z-z0,r_length,vx,vy,vz,v_length,phi)
    data = np.array(values,dtype)
    df = pd.DataFrame(data = data, columns = ['x','y','z','r_length','vx','vy','vz','v_length','phi'])
    #print df['x']
    df = df.sort_values(by=['r_length'])
    #print df['r_length']
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    r = df[['x','y','z']].values
    r_length = df['r_length'].values
    vx = df['vx'].values
    vy = df['vy'].values
    vz = df['vz'].values
    v = df[['vx','vy','vz']].values
    v_length = df['v_length'].values
    phi = df['phi'].values

################# calculate mass enclosed   ##############
time2 = time.time()
if radius_sort == 'True':
    # if particles are sorted by distance from center, the mass enclosed is just the addition of one particle per step
    mass_enclosed = np.arange(1,len(r_length)+1)*massp*10
else:
    mass_enclosed = np.zeros(len(r_length))
    print 'calculating mass enclosed'
    #for i in range(0,len(r_length)):
    #    if i % 10000 == 0:
    #        print i,'out of',len(r_length), 'particles'
    #    q = r_length[r_length<=r_length[i]]
    #    mq = float(len(q))
    #    mass_enclosed[i] = mq*massp*10.
    #    q = []
    
    # determining mass with parallel set up    
    def massfunc(i,r_length,massp):
        q = r_length[r_length<=r_length[i]]
        mq = float(len(q))
        mass_enclosed = mq*massp*10.
        return mass_enclosed
    mass_enclosed = Parallel(n_jobs=num_cores)(delayed(massfunc)(i,r_length,massp) for i in range(0,len(r_length)))

#mass_enclosed = np.arange(1,len(r_length)+1)*massp*10
mtot = float(len(r_length))*massp*10.
#print mass_enclosed,np.sort(mass_enclosed),mtot

#checks to see if mass in correct order
if checkmass == 'True':
    masse = np.sort(mass_enclosed)
    re = np.sort(r_length)
    print 'Mass ordering is:', np.allclose(mass_enclosed,masse)
    print 'Radius ordering is', np.allclose(r_length,re)
#print mtot, max(r_length)
time3 = time.time()


################### calculate angular momentum and energy per mass ########################

if radius_sort == 'True':
    L = np.cross(r,v)
else:
    L = np.cross(r.T,v.T)

#print L, sum(L)
#print r
#print v

## calculate L^2
L2 = np.zeros(len(L))
LN = np.zeros(len(L))
for i in range(len(L)):
    L2[i] = np.dot(L[i,:],L[i,:])
    LN[i] = np.linalg.norm(L[i,:])
#print L2.shape, L2

#print v[1,:],v.T[1,:]
E = np.zeros(len(L))
for i in range(len(L)):
    if radius_sort == 'True':
        E[i] = 1./2.*np.dot(v[i,:],v[i,:]) + phi[i]
    else:
        E[i] = 1./2.*np.dot(v.T[i,:],v.T[i,:]) + phi[i]
print 'Min E=', min(E), 'Max E=',max(E)#,E[0],E[1]
#print E, sum(E)

# remove unbound particles
if bound=='True':
    Eub = E[E>0]
    E = E[E<0]
    mass_enclosed = mass_enclosed[E<0]
    r_length = r_length[E<0]
    L2 = L2[E<0]
    phi = phi[E<0]
    print 'Number of unbound particles:',len(Eub)


#############angular momentum optimum in circular orbit with a given r
r_circ = np.sort(r_length)
v_circ = np.sqrt(G*np.sort(mass_enclosed)/r_circ)
v_circ[0] = 0
L_circ = r_circ*v_circ
#print L_circ
#print L
L2_circ = L_circ**2.
#L2_circ = np.sort(L2_circ)

print 'L2_circ is',L2_circ
#print v_circ
#print v_length



###########calculate potential energy#####################################

mass_enclosed_sort = np.sort(mass_enclosed)
r_length_sort = np.sort(r_length)

# calculate density profile, assume radial bins with constant density inside, bin width = softening parameter


print 'calculating density profile'
nebins = r_vir/(sft) #define number of radial bins
step = sft #size of each bin
#define bins across radius in pc
rebins = np.arange(step,r_vir+step,step)#in km
#define centers of bins
repoints = np.arange(step/2.,r_vir+step/2.,step)#in km
rhob = np.zeros(len(r_length_sort))
pot_circ = np.zeros(len(r_length))
rj = r_length_sort.tolist()
for i in range(1,len(rebins)):
    count = 0
    massinbin = 0
    if i%20. == 0:
        print i,'out of',int(nebins), 'radial bins'
    for j in range(len(r_length_sort)):
        if rebins[i-1]<r_length_sort[j]<rebins[i]:
            count = count+1
                #print j
        massinbin = count*massp*10
        rhoinbin = massinbin/(4.*np.pi/3.*(rebins[i]**3-rebins[i-1]**3))
    for j in range(len(r_length_sort)):
        if rebins[i-1]<r_length_sort[j]<rebins[i]:
            rhob[j] = rhoinbin
    #for j in rj:
    #    if rebins[i-1]<j<rebins[i]:
    #        #if r_length_sort[j]>rebins[i-1]:
    #        count = count+1
    #            #print j
    #    massinbin = count*massp*10
    #    rhoinbin = massinbin/(4.*np.pi/3.*(rebins[i]**3-rebins[i-1]**3))
    #for j in rj:
    #    if rebins[i-1]<j<rebins[i]:
    #        #if r_length_sort[j]>rebins[i-1]:
    #        rhob[rj.index(j)] = rhoinbin
    #        #rhob[np.where(rj==j)] = rhoinbin
    #        rj.remove(j)
    #        #rj = np.delete(rj,j)
            
                   
            
            
    #def rhofunc(i,j,rebins,r_length_sort,rhoinbin):
    #    if rebins[i-1]<r_length_sort[j]<rebins[i]:
    #        #if r_length_sort[j]>rebins[i-1]:
    #        #rhobb = rhoinbin
    #        return rhoinbin 
    #rhob = Parallel(n_jobs=num_cores)(delayed(rhofunc)(i,j,rebins,r_length_sort,rhoinbin) for j in range(len(r_length_sort)))
    
    #print rhob[j]

time4 = time.time()
print 'calculating potentials'
#for k in range(1,len(r_length_sort)):
#    if k % 10000 == 0:
#        print k,'out of',len(r_length), 'particles'
    #pot_circ = -4.*np.pi*G*(1/r_length_sort[k]*scipy.integrate.simps(r_length_sort[:k+1]**2*rhob[:k+1],r_length_sort[:k+1])+scipy.integrate.simps(r_length_sort[k:]*rhob[k:],r_length_sort[k:]))
#pot = np.zeros(len(r_length_sort))    
def potfunc(k,r_length_sort,rhob,G):
    pott = -4.*np.pi*G*( 1./r_length_sort[k]*trapz(r_length_sort[:k+1]**2*rhob[:k+1],r_length_sort[:k+1])+trapz(r_length_sort[k:]*rhob[k:],r_length_sort[k:]))
    #pott = -4.*np.pi*G*( 1./r_length_sort[k]*simps(r_length_sort[:k+1]**2*rhob[:k+1],r_length_sort[:k+1])+simps(r_length_sort[k:]*rhob[k:],r_length_sort[k:]))
    return np.array(pott)
        
#pot = Parallel(n_jobs=num_cores)(delayed(potfunc)(k,r_length_sort,rhob) for k in range(1,len(r_length_sort)))
#print pot
pot_circ = Parallel(n_jobs=num_cores)(delayed(potfunc)(k,r_length_sort,rhob, G) for k in range(1,len(r_length_sort)))
pot_circ.insert(0,0)

time5 = time.time()

#print pot_circ
#print rhob
#plt.figure()
#plt.scatter(r_length_sort,rhob)
#plt.figure()
#plt.scatter(r_length_sort,pot_circ)
#plt.show()
#exit()


#t = (1,2,3.5,4,5)
#print t, t[3:]
#exit()


#### add potential and kinetic
E_circ = 0.5*((v_circ))**2+pot_circ

E2 = np.zeros(len(L))
for i in range(len(L)):
    E2[i] = 1./2.*np.dot(v.T[i,:],v.T[i,:]) + pot_circ[i]
 



print E_circ#, r_length
print np.sort(L2_circ)#, r_length, mass_enclosed
print np.sort(L2)
#print len(E_circ),len(L2_circ)

print 'total',time5-time1
print 'mass enclosed', time3-time2
print 'potentials', time5-time4



#############################plots for data########################################

plt.figure()
plt.scatter(r_length,L2,color='red')
plt.scatter(r_circ,L2_circ,color='blue')
plt.xlabel("Radius (km)")
plt.ylabel("Angular Momentum $(km*km/s)^2$")

plt.figure()
plt.scatter(r_circ,v_circ,color='blue')
plt.xlabel("Radius (km)")
plt.ylabel("Circular Velocity (km/s)")



if supress_circle_analysis == 'False':
    '''
    plt.figure()
    plt.scatter(r_circe,E)
    plt.xlabel('Radius (km)')
    plt.ylabel('E')
    plt.title('Given E')
    #'''
    plt.figure()
    plt.scatter(r_length_sort[1:],E_circ[1:])
    plt.xlabel('Radius (km)')
    plt.ylabel('E')
    plt.title('Given r')
    
    plt.figure()
    plt.scatter(r_length_sort[1:],pot_circ[1:])
    plt.xlabel('Radius (km)')
    plt.ylabel('Potenial E')
    plt.title('Given r')

if supress_mass_enclosed == 'False':
    plt.figure()
    plt.scatter(r_length,mass_enclosed)
    plt.title('Mass Enclosed')

#anisotropy plots
if supress_velocity_anisotropy == 'False':
    plt.figure()
    plt.plot(rpoints/pc,beta)
    plt.xlabel('Radius (Mpc)')
    plt.ylabel(r'$\beta$')

    plt.figure()
    plt.plot(rpoints/pc,vr_ave)
    plt.plot(rpoints/pc,vtheta_ave)
    plt.plot(rpoints/pc,vphi_ave)
    plt.legend(('vr','vtheta','vphi'))
    plt.xlabel('Radius (Mpc)')
    plt.ylabel('Average velocity (km/s)')

    plt.figure()
    plt.plot(rpoints/pc,sigma_r)
    plt.plot(rpoints/pc,sigma_theta)
    plt.plot(rpoints/pc,sigma_phi)
    #plt.legend((r'$\simga_r$',r'$\simga_{\theta}$',r'$\simga_{\phi}$'))
    plt.legend(('simga_r','simga_theta','simga_phi'))
    plt.xlabel('Radius (Mpc)')
    plt.ylabel('Velocity dispersion $\sigma$ (km/s)')




#plt.figure()
#plt.plot(r_length,phi_r)



if supress_EL2 == 'False':
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(E,np.log10(L2))
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('Log $(L^2)$ $(km*km/s)^2$') 
    plt.xlim((min(E), max(E)))
    #plt.ylim((40, max(np.log10(L2))))
    plt.title('Data')
    plt.subplot(1, 2, 2)
    plt.scatter(E,(L2))
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('$L^2$ $(km*km/s)^2$') 
    plt.xlim((min(E), max(E)))
    #plt.ylim((40, max(np.log10(L2))))
    plt.title('Data')


    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(E_circ[1:],np.log10(L2_circ[1:]))
    plt.title('L2_circ')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('Log $(L^2)$ $(km*km/s)^2$')
    plt.xlim((min(E_circ)), (max(E_circ)))
    #plt.ylim(25., 60)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(E_circ[1:],(L2_circ[1:]))
    plt.title('L2_circ')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('$(L^2)$ $(km*km/s)^2$')
    plt.xlim((min(E_circ)), (max(E_circ)))
    #plt.ylim(25., 60)
    plt.grid(True)

    '''
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(E,np.log10(L2_circe))
    plt.title('L2_circe')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('Log $(L^2)$ $(km*km/s)^2$')
    plt.xlim((min(E), max(E)))
    #plt.ylim(40., 60)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(E,(L2_circe))
    plt.title('L2_circe')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('$(L^2)$ $(km*km/s)^2$')
    plt.xlim((min(E), max(E)))
    #plt.ylim(40., 60)
    plt.grid(True)
    #'''

    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(E,np.log10(L2))
    plt.scatter(E_circ,np.log10(L2_circ), color='green')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('$L^2$ $(km*km/s)^2$') 
    plt.xlim((min(E), max(E)))
    #plt.ylim((40, max(np.log10(L2))))
    plt.title('Data')
    plt.subplot(1,2,2)
    plt.scatter(E,(L2))
    plt.scatter(E_circ,(L2_circ), color='green')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('$L^2$ $(km*km/s)^2$') 
    plt.xlim((min(E), max(E)))
    #plt.ylim((40, max(np.log10(L2))))
    plt.title('Data')
    
    plt.figure()
    plt.scatter(E,np.log10(LN))
    plt.scatter(E_circ,np.log10(L_circ), color='green')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('Log(L) $(km*km/s)^2$') 
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(E2,np.log10(L2))
    plt.scatter(E_circ,np.log10(L2_circ), color='green')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('$L^2$ $(km*km/s)^2$') 
    plt.xlim((min(E), max(E)))
    #plt.ylim((40, max(np.log10(L2))))
    plt.title('Data')
    plt.subplot(1,2,2)
    plt.scatter(E2,(L2))
    plt.scatter(E_circ,(L2_circ), color='green')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('$L^2$ $(km*km/s)^2$') 
    plt.xlim((min(E), max(E)))
    #plt.ylim((40, max(np.log10(L2))))
    plt.title('Data')
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(E2[E2<0],np.log10(L2)[E2<0])
    plt.scatter(E_circ,np.log10(L2_circ), color='green')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('$L^2$ $(km*km/s)^2$') 
    plt.xlim((min(E), max(E)))
    #plt.ylim((40, max(np.log10(L2))))
    plt.title('Data')
    plt.subplot(1,2,2)
    plt.scatter(E2[E2<0],(L2)[E2<0])
    plt.scatter(E_circ,(L2_circ), color='green')
    plt.xlabel('$E$ $(km/s)^2$')
    plt.ylabel('$L^2$ $(km*km/s)^2$') 
    plt.xlim((min(E), max(E)))
    #plt.ylim((40, max(np.log10(L2))))
    plt.title('Data E>0')

#calculate data for 2D histogram
'''
fig = plt.figure()
ax = fig.add_subplot(111)
H, xedges, yedges = np.histogram2d(L2, E, bins=(1000, 100))
extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
levels = (1.0e4, 1.0e3, 1.0e2, 2.0e1)
cset = contour(H, levels, origin='lower',colors=['black','green','blue','red'],linewidths=(1.9, 1.6, 1.5, 1.4),extent=extent)
plt.clabel(cset, inline=1, fontsize=10, fmt='%1.0i')
for c in cset.collections:
    c.set_linestyle('solid')
#'''




if supress_2Dhist == 'False':
    L22 = np.log10(L2)
    L22[0] = 1.
    #L22=L22.T
    #bins = [10 ** np.linspace(np.log10(min(E)), np.log10(max(E)), 1000), 10 ** np.linspace(-2, np.log10(max(L2)), 1000)]
    #H, xedges, yedges = np.histogram2d(L2[E<-750000],E[E<-750000], bins=[10000,100])
    H, xedges, yedges = np.histogram2d(L2,E, bins=[10000,100])



    extent = [ yedges[0], yedges[len(yedges)-1], xedges[0], xedges[len(xedges)-1] ]
    # Find limits of histogram to scale colorbar correctly
    vmin = np.min(H[H!=0])
    vmax = np.max(H)

    plt.figure()
    plt.imshow(H,aspect='auto',cmap=plt.cm.jet,vmin=vmin, vmax=vmax, extent=extent, origin='lower', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Energy")
    plt.ylabel("$L^2$")

#print len(E[E>-600000]), len(L2[L2>100000])

if supress_circleEL2 == 'False':

    Ebins = 10
    H, xedges, yedges = np.histogram2d(L2,E, bins=[10000,Ebins])
    extent = [ yedges[0], yedges[len(yedges)-1], xedges[0], xedges[len(xedges)-1] ]
    # Find limits of histogram to scale colorbar correctly
    vmin = np.min(H[H!=0])
    vmax = np.max(H)
    #print H[0,:]
    Ebins_center = (yedges[:-1]+yedges[1:])/2
    #print 'Energy center', Ebins_center


    ############# for actual particles
    #print yedges, len(yedges), max(E), min(E)
    L2bf = np.zeros(len(yedges))
    chi2f = np.zeros(len(yedges))
    p_val = np.zeros(len(yedges))
    for j in range(0,len(yedges)):
        L2b = []
        for i in range(1,len(E)):
            if E[i]<yedges[j]:
                if E[i]>yedges[j-1]:
                    L2b = np.append(L2b,L2[i])
        #print L2b, len(L2b)  
        #L2bf[j] = L2b
        if supress_EL2_histograms == 'False':
            if j % 5 == 0:
                if L2b != []:
                    print j,'out of',Ebins,'Ebins'#,L2b, len(L2b)
                    plt.figure()
                    plt.hist(L2b,bins=100)
                    plt.xlabel('$L^2$ $(km*km/s)^2$')
                    plt.xlabel('$(L/L_{circ})^2$')
                    plt.ylabel('Counts (C_tot=%s)'%(len(L2b)))
                    plt.title('Energy bin %s which is for %s<E<%s'%(j, yedges[j-1],yedges[j]))
            if j == 8:
                plt.figure()
                plt.hist(L2b,bins=100)
                plt.xlabel('$L^2$ $(km*km/s)^2$')
                plt.ylabel('Counts (C_tot=%s)'%(len(L2b)))
                plt.title('Energy bin %s which is for %s<E<%s'%(j, yedges[j-1],yedges[j]))
                
                
        if len(L2b)>100:
            data,nothing = np.histogram(L2b,bins=20)
            data = np.array(data)
            chi2f[j],p_val[j] = chi2(data)
    
    print 'Chi^2 for Ebins (0 if unpopulated or not enough particles)',chi2f
    #print 'Chi^2 for Ebins (0 if unpopulated or not enough particles)',p_val
    #exit()
    
    
    ################## for circular orbit particles###################
    '''
    L2b = []
    for i in range(1,len(E)):
        if E[i]<-1200000:   #-7.36e12 with mass !=1
            if E[i]>-1250000: #-7.49e12 with mass not =1
                L2b = np.append(L2b,L2_circ[i])
    #print L2b, len(L2b)          
    plt.figure()
    plt.hist(L2b,bins=20)
    plt.xlabel("$L_{circ}^2 $(km*km/s)^2$")
    plt.ylabel('Counts')
    #'''



if supress_basic_histograms == 'False':
    plt.figure()
    plt.hist(L2,bins=50)
    plt.xlabel('$L^2$ $(km*km/s)^2$')

    plt.figure()
    plt.hist(E,bins=50)
    plt.xlabel('$E$ $(km/s)^2$')


    plt.figure()
    if cutoff == 'True':
        plt.hist(r_length[cond2]/pc,bins=50)
    else:
        plt.hist(r_length/pc,bins=50)
    plt.xlabel('R (Mpc)')






plt.show()

'''
############## save output files##################################
out = np.row_stack(L)
np.savetxt('angularm.txt',out,fmt='%.9f')
out = np.row_stack(L2)
np.savetxt('angularm_squared.txt',out,fmt='%.9f')
out = np.row_stack(E)
np.savetxt('totalE.txt',out,fmt='%.9f')
#'''


