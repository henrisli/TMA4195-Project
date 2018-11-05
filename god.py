
import numpy as np

def god(h0, cfl, dx, T, flux, dflux, boundary, production):
    def lim(r):
        return np.maximum(0, np.minimum(1.3*r, np.minimum(0.5+0.5*r, 1.3)))
    
    def limiter(a,b):
        return lim(np.divide(b, a + 1e-6))*a
    
    dt = cfl*dx/max(abs(dflux(h0)))
    h = np.copy(h0)
    H = np.copy(h0)
    t = 0.0
    n = len(h0)
    f = np.zeros(n)
    i = np.arange(2,n-2)
    j = np.arange(n-1)
    phi = np.zeros(n)
    for k in range(T):
        if (t+dt > T):
            dt = T-t
        
        t += dt
        h = boundary(h, 2)
        q = production(h)
        dh = np.diff(h)
        phi[1:-1] = limiter(dh[:-1], dh[1:]) 
        phi[0] = phi[-2]
        phi[-1] = phi[1]  
        hr = h + 0.5*phi
        fr = flux(hr)
        dfr = dflux(hr)
        mdf = max(dfr)
        f[j] = fr[j]
        H[i] = h[i] - dt/dx*(f[i]-f[i-1]) + dt*q[i]
          
        H = boundary(H,2)
        Q = production(H)
        dh = np.diff(H)
        phi[1:-1] = limiter(dh[:-1], dh[1:])
        phi[0] = phi[-2]
        phi[-1] = phi[1]  
        hr = H + 0.5*phi
        fr = flux(hr)
        dfr = dflux(hr)
        mdf = np.maximum(dfr, mdf)
        mdf = max(mdf)
        f[j] = fr[j]
        h[i] = 0.5*h[i] + 0.5*( H[i]-dt/dx*(f[i]-f[i-1]) ) + dt*Q[i]
        dt = cfl*dx/mdf
    h = boundary(h, 2)
    dh = np.diff(H)
    phi[1:-1] = limiter(dh[:-1], dh[1:])
    phi[0] = phi[-2]
    phi[-1] = phi[1]
    return h, phi