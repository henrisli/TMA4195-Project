
import numpy as np


def upw(h0, cfl, dx, T, flux, df, boundary, production):
    h = np.copy(h0)
    t = 0.0
    dt = cfl*dx/max(abs(df(h0)))
    i = np.arange(1,len(h0)-1,1)
    k = 0
    while t<T:
        if t+dt > T:
            dt = T-t
        t += dt
        h = boundary(h)  
        f = flux(h, dx)
        q = production(h,k)
        h[i] = h[i] - dt/dx*(f[i]-f[i-1]) + dt*q[i]
        dt = cfl*dx/max(abs(df(h)))
        k +=1
    return h