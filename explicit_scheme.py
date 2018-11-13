import numpy as np
from diffusion import diffusion
import matplotlib.pyplot as plt

def explicit_scheme(dx,K,H0,dt,tf,production,d,boundary):
    H = 100
    L = 6000
    Q = 2/(365*24*3600)
    mu = 9.3e-25
    m = 3
    rho = 1000
    g = 9.81
    alpha_s = 3*np.pi/180
    Theta_s = rho*g*H*np.sin(alpha_s)
    kappa_s = 2*H**2/(Q*L)*mu*Theta_s**m
    Gamma = kappa_s/(m+2)
    gamma = H/(L*np.tan(alpha_s))
    H = np.copy(H0)
    k = np.arange(1,K+1,1)
    ek = np.arange(2,K+2,1)
    wk = np.arange(0,K,1)
    t = 0.0
    dtlist = []
    j = 0
    while t < tf:
        if t+dt > tf:
            dt = tf-t
        H = boundary(H)
        Hrt = 0.5*(H[ek] + H[k])
        Hlt = 0.5*(H[k] + H[wk])
        a2rt = np.power(np.abs(gamma*(np.diff(H)[1:])/dx-1),m-1)
        a2lt = np.power(np.abs(gamma*(np.diff(H)[:-1])/dx-1),m-1)
        Drt = Gamma * np.power(Hrt,m+2) * a2rt
        Dlt = Gamma * np.power(Hlt,m+2) * a2lt
        #H,dtadapt,q_test = diffusion(dx,K,Drt,Dlt,H,dt,production,d,gamma,j)
        mu_x = dt/(dx*dx)
        Hb = H + d
        q_p = production(H,j)
        H[1:-1] = H[1:-1] + mu_x*Drt*(gamma*(Hb[2:] - Hb[1:-1])-dx) - mu_x*Dlt*(gamma*(Hb[1:-1]-Hb[:-2])-dx)+ dt*q_p[1:-1]
        H[H<1e-06] = 0
        maxD = [max(Dlt), max(Drt)]
        maxD = max(maxD)
        t += dt
        dtlist = np.append(dtlist,dt)
        j += 1
    print(j)
    print(maxD)
    plt.figure()
    plt.plot(q_p)
    return H, dtlist
