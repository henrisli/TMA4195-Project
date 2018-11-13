
import numpy as np


def upw(h0, cfl, dx, T, flux, df, boundary, production, d):
    h = np.copy(h0)
    t = 0.0
    dt = cfl*dx/max(abs(df(h0)))
    i = np.arange(1,len(h0)-1,1)
    for k in range(T):
        t += dt
        h = boundary(h)  
        f = flux(h, d, dx)
        q = production(h,k,d)
        h[i] = h[i] - dt/dx*(f[i]-f[i-1]) + dt*q[i]
        dt = cfl*dx/max(abs(df(h)))
    return h,t