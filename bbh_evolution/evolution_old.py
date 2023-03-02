import numpy as np
import constants as const
import auxiliary as auxi

#constants
#G=6.6743015e-8  #gravity constant cgs
#c=2.99792458e10 #speed of light cgs

G3=const.G**3.
c5=const.c**5.

#yr=3.1557600e7 #yr/seconds
 
#msun=1.98892e33 #msun/g
#AU=1.49598073e+13 #AU/cm
#Rsun=6.95700e10 #rsun/cm
 
#tolerance factor: abs(a-anew)/a ~ tol
tol=1e-2

def ode_GWonly(m1,m2,a,e): #peters 1964 formulas
    #print(e)
    if(e<0.0):
        print("horror, I have neg e = ",e)
    #    e=0.0
    if(a<0.0):
        print("horror, I have neg a = ",a)
    #    a=1e-20        
    fa= -64./5.* G3 * m1 * m2 * (m1+m2)/(c5 * a*a*a * (1-e*e)**3.5) * (1.+73./24.*e*e+37./96. * e * e * e * e)
    fe=-304./15. * e * G3 * m1 * m2 * (m1+m2)/(c5 * a*a*a*a * (1-e*e)**2.5) * (1.+121./304. * e * e)
    return (fa,fe) #outputs are da/dt, de/dt (peters 1964)


def euler_GWonly(m1,m2,a,e,h):
    fa,fe=ode_GWonly(m1,m2,a,e)
    a=a+fa*h
    e=e+fe*h
    return a,e



def ode(m1,m2,a,e,xi,ki,rho,sigma): #peters 1964 formulas + hardening
    #print(e)
    if(e<0.0):
        print("horror, I have neg e = ",e)
    #    e=0.0
    if(a<0.0):
        print("horror, I have neg a = ",a)
    #    a=1e-20

    fa= - 2. * np.pi * xi * const.G * rho/sigma * a* a \
        -64./5.* G3 * m1 * m2 * (m1+m2)/(c5 * a*a*a * (1-e*e)**3.5) * (1.+73./24.*e*e+37./96. * e * e * e * e)
    fe= + ki * 2. * np.pi * xi * const.G * rho/sigma * a \
        -304./15. * e * G3 * m1 * m2 * (m1+m2)/(c5 * a*a*a*a * (1-e*e)**2.5) * (1.+121./304. * e * e) 
    #if( 2. * np.pi * xi * const.G * rho/sigma * a* a>64./5.* G3 * m1 * m2 * (m1+m2)/(c5 * a*a*a * (1-e*e)**3.5) * (1.+73./24.*e*e+37./96. * e * e * e * e)):
    #    print(m1,"fe = ", ki * 2. * np.pi * xi * const.G * rho/sigma * a, -304./15. * e * G3 * m1 * m2 * (m1+m2)/(c5 * a*a*a*a * (1-e*e)**2.5) * (1.+121./304. * e * e))
    #    print(m1,"fa = ",- 2. * np.pi * xi * const.G * rho/sigma * a* a,-64./5.* G3 * m1 * m2 * (m1+m2)/(c5 * a*a*a * (1-e*e)**3.5) * (1.+73./24.*e*e+37./96. * e * e * e * e))
    return (fa,fe) #outputs are da/dt, de/dt (peters 1964)

def euler(m1,m2,a,e,h,xi,ki,rho,sigma):
    fa,fe=ode(m1,m2,a,e,xi,ki,rho,sigma)
    a=a+fa*h
    e=e+fe*h
    return a,e


##################MAIN ROUTINE#####################

def peters_evolv(BBH, SC, i, t3bb, vesc0, rh0, aej, agw, flags):
    #contains preliminary star cluster evolution, do not consider it
    
    m1 = BBH.m1[i]
    m2 = BBH.m2[i]
    a =  BBH.sma[i]
    e =  BBH.ecc[i]

    sigma = SC.sigma[i]
    SClifetime = SC.SClifetime[i]
    vesc = SC.vesc[i]
    rh = SC.rh[i]
    trh0 = SC.trh0[i]
    Mtot = SC.Mtot[i]
    maverage = SC.maverage[i]
    
    xi = BBH.xi
    ki = BBH.ki


    
    #go cgs
    m1 = m1 * const.msun
    m2 = m2 * const.msun
    a = a * const.Rsun
    #print(t3bb,tH)
    tbb=t3bb * 1e6 * const.yr
    tH=const.tHubble * 1e6 * const.yr
    trh0cgs=trh0 * 1e6 * const.yr
    SClife=SClifetime * 1e6 * const.yr

    tapprox=5./256.*c5/G3*a*a*a*a * (1.-e*e)**3.5/m1/m2/(m1+m2)

    anew=0.0
    enew=0.0

    ecc10=100.0 #initialize eccentricity at 10 Hz to crazy value
    freqold=-10.0

    h=0.1*tapprox
    t=0.0
    rth=3.*2.*const.G*(m1+m2)/const.c/const.c #Schw. radius of bh in cm

    vesc,sigma,rh,rho=auxi.update_vesc_rh_rho(vesc0,rh0,trh0cgs,Mtot,(t+tbb))

    while(abs(a-rth)/rth>1e-1):
        if(((t+tbb)<=SClife) and ((aej<=agw) or ((aej>agw) and (a>=aej))) and (flags.flagSN[i]=="non_ejected_by_SN")):
            flag_evap="in_cluster"

            vesc,sigma,rh,rho=auxi.update_vesc_rh_rho(vesc0,rh0,trh0cgs,Mtot,(t+tbb))
            #rho=rho*const.msun/const.parsec**3
            rhoc=const.c2hm_dens * rho * const.msun/const.parsec**3 #use the core density instead of half-mass density here - consider using 4e6 (sigma/100 kms)**2 from antonini&rasio 2016

            aej=auxi.a_ej(xi,maverage,m1/const.msun,m2/const.msun,vesc) 
            aGW=auxi.a_GW(e,m1/const.msun,m2/const.msun,xi,sigma,rhoc) #think about sigma!!!
        
            anew,enew=euler(m1,m2,a,e,h,xi,ki,rhoc,sigma)
            if(abs(anew-a)/a<(0.1*tol)): #set adaptive timestep
                h=h*2.
                anew,enew=euler(m1,m2,a,e,h,xi,ki,rhoc,sigma)
            
            elif(abs(anew-a)/a>tol):
                while(abs(anew-a)/a>tol):
                    h=h/10.
                    anew,enew=euler(m1,m2,a,e,h,xi,ki,rhoc,sigma)
            a=anew
            e=enew
            
            freqGW=1./np.pi * np.sqrt(const.G*(m1+m2)/a**3)
            if(abs(freqGW-10.0)<abs(freqold-10.0)): #frequency of GW = 10 Hz
                ecc10=e
                freqold=freqGW

            if((t+tbb)>tH):
                flag="no_merg"
                #print(flag,(t+tbb)/(yr*1e9))
                break
            else:
                flag="merg"
            t+=h

        else:
            if(flags.flagSN[i]=="ejected_by_SN"):
                flag_evap="ejected_by_SN"
            else:
                if((t+tbb)>SClife):
                    flag_evap="cluster_died"
                else:
                    if(aej>agw):
                        flag_evap="ejected_3B"
                    elif(aej<=agw):
                        flag_evap="in_cluster"
                        print("it should never happen\n")

            anew,enew=euler_GWonly(m1,m2,a,e,h)
            if(abs(anew-a)/a<(0.1*tol)): #set adaptive timestep
                h=h*2.
                anew,enew=euler_GWonly(m1,m2,a,e,h)
            
            elif(abs(anew-a)/a>tol):
                while(abs(anew-a)/a>tol):
                    h=h/10.
                    anew,enew=euler_GWonly(m1,m2,a,e,h)
            a=anew
            e=enew

            freqGW=1./np.pi * np.sqrt(const.G*(m1+m2)/a**3)
            if(abs(freqGW-10.0)<abs(freqold-10.0)): #frequency of GW = 10 Hz
                ecc10=e
                freqold=freqGW


            if((t+tbb)>tH):
                flag="no_merg"
                #print(flag,(t+tbb)/(yr*1e9))
                break
            else:
                flag="merg"
            t+=h

    #rho=rho*const.parsec**3/const.msun
    
    BBH.sma2[i] = a
    BBH.ecc2[i] = e
    BBH.ecc10[i] = ecc10
    
    SC.vesc[i] = vesc
    SC.sigma[i] = sigma
    SC.rh[i] = rh
    SC.rho[i] = rho

    flags.flag2[i] = flag
    flags.flag_evap[i] = flag_evap

    
    return(BBH, SC,t,flags)


def peters(BBH, SC, i, t3bb, aej, agw, flags):
    #USE IT! version without star cluster evolution
    m1 = BBH.m1[i]
    m2 = BBH.m2[i]
    a =  BBH.sma[i]
    e =  BBH.ecc[i]
    
    rho = SC.rhocgs[i]
    sigma = SC.sigma[i]
    SClifetime = SC.SClifetime[i]

    xi = BBH.xi
    ki = BBH.ki
    
    
    #go cgs
    m1 = m1 * const.msun
    m2 = m2 * const.msun
    a = a* const.Rsun
    #print(t3bb,tH)
    tbb=np.copy(t3bb) * 1e6 * const.yr
    tH=const.tHubble * 1e6 * const.yr
    SClife=SClifetime * 1e6 * const.yr

    rhoc= const.c2hm_dens * rho #use the core density instead of half-mass density here    

    tapprox=5./256.*c5/G3*a*a*a*a * (1.-e*e)**3.5/m1/m2/(m1+m2)

    anew=0.0
    enew=0.0
    ecc10=100.0 #initialize eccentricity at 10 Hz to crazy value
    freqold=-10.0

    h=0.1*tapprox
    t=0.0
    rth=3.*2.*const.G*(m1+m2)/const.c/const.c #Schw. radius of bh in cm
    
    
    
    while(abs(a-rth)/rth>1e-1):
        if(((t+tbb)<=SClife) and ((aej<=agw) or ((aej>agw) and (a>=aej))) and (flags.flagSN[i]=="non_ejected_by_SN")):

            #if(((t+tbb)<=SClife) and (a>=aej)):
            flag_evap="in_cluster"

            anew,enew=euler(m1,m2,a,e,h,xi,ki,rhoc,sigma)
            if(abs(anew-a)/a<(0.1*tol)): #set adaptive timestep
                h=h*2.
                anew,enew=euler(m1,m2,a,e,h,xi,ki,rhoc,sigma)
            
            elif(abs(anew-a)/a>tol):
                while(abs(anew-a)/a>tol):
                    h=h/10.
                    anew,enew=euler(m1,m2,a,e,h,xi,ki,rhoc,sigma)
            a=anew
            e=enew

            freqGW=1./np.pi * np.sqrt(const.G*(m1+m2)/a**3)
            if(abs(freqGW-10.0)<abs(freqold-10.0)): #frequency of GW = 10 Hz
                ecc10=e
                freqold=freqGW

            if((t+tbb)>tH):
                flag="no_merg"
                #print(flag,(t+tbb)/(yr*1e9))
                break
            else:
                flag="merg"
            t+=h
        else:
            if(flags.flagSN[i]=="ejected_by_SN"):
                flag_evap="ejected_by_SN"
                
            else:
                if((t+tbb)>SClife):
                    flag_evap="cluster_died"
                else:
                    if(aej>agw):
                        flag_evap="ejected_3B"
                    elif(aej<=agw):
                        flag_evap="in_cluster"
                        print("it should never happen\n")

            anew,enew=euler_GWonly(m1,m2,a,e,h)
            if(abs(anew-a)/a<(0.1*tol)): #set adaptive timestep
                h=h*2.
                anew,enew=euler_GWonly(m1,m2,a,e,h)
            
            elif(abs(anew-a)/a>tol):
                while(abs(anew-a)/a>tol):
                    h=h/10.
                    anew,enew=euler_GWonly(m1,m2,a,e,h)
            a=anew
            e=enew

            freqGW=1./np.pi * np.sqrt(const.G*(m1+m2)/a**3)
            if(abs(freqGW-10.0)<abs(freqold-10.0)): #frequency of GW = 10 Hz
                ecc10=e
                freqold=freqGW

            if((t+tbb)>tH):
                flag="no_merg"
                #print(flag,(t+tbb)/(yr*1e9))
                break
            else:
                flag="merg"
            t+=h

    BBH.sma2[i] = a
    BBH.ecc2[i] = e
    BBH.ecc10[i] = ecc10

    flags.flag2[i] = flag
    flags.flag_evap[i] = flag_evap

    return(BBH, t, flags)
