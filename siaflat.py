import numpy as np
from diffusion import diffusion

def siaflat(Lx,K,H0,deltat,tf,q,d):
#    g = 9.81
#    rho = 910.0
#    secpera = 31556926
#    A = 1.0e-12/secpera
#    Gamma = 2 * A * (rho * g)**3 / 5
    H = 50
    L = 2000
    Q = 3.5/(365*24*3600)
    mu = 9.3e-25
    m = 3
    rho = 1000
    g = 9.81
    alpha_s = 3*np.pi/180
    Theta_s = rho*g*H*np.sin(alpha_s)
    kappa_s = 2*H**2/(Q*L)*mu*Theta_s**m
    Gamma = kappa_s/(m+2)
    gamma = H/(L*np.tan(alpha_s))
    print(gamma)
    H = H0
    dx = 2 * Lx / K
    N = int(np.rint(tf / deltat))
    deltat = tf / N
    k = np.arange(1,K+1,1)
    ek = np.arange(2,K+2,1)
    wk = np.arange(0,K,1)
    t = 0
    dtlist = []
    for n in range(N):
        print(n)
        Hrt = 0.5*(H[ek] + H[k])
        Hlt = 0.5*(H[k] + H[wk])
        a2rt = np.power(gamma*(H[ek] - H[k])-1,m-1) / (dx**2)
        a2lt = np.power(gamma*(H[k] - H[wk])-1,m-1) / (dx**2)
        Drt = Gamma * np.power(Hrt,m+2) * a2rt
        Dlt = Gamma * np.power(Hlt,m+2) * a2lt
        H,dtadapt = diffusion(Lx,K,Drt,Dlt,H,deltat,q,d)
        t += deltat
        dtlist = np.append(dtlist,dtadapt)
    return H, dtlist
