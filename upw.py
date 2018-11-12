
import numpy as np
import matplotlib.pyplot as plt

def upw(h0, cfl, dx, T, flux, df, boundary, production, d):
    h = np.copy(h0)
    t = 0.0
    dt = cfl*dx/max(abs(df(h0)))
    i = np.arange(1,len(h0)-1,1)
    for k in range(T):
        t += dt
        h = boundary(h)  
        f = flux(h, d, dx)
        q = production(h,k)
        h[i] = h[i] - dt/dx*(f[i]-f[i-1]) + dt*q[i]
        h[h<1e-16] = 0
        dt = cfl*dx/max(abs(df(h)))
    return h,t