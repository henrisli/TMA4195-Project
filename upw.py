
import numpy as np
import matplotlib.pyplot as plt

def upw(h0, cfl, dx, T, flux, df, boundary, production, d):
    h = np.copy(h0)
    t = 0.0
    dt = cfl*dx/max(abs(df(h0)))
    i = np.arange(1,len(h0)-1,1)
    j = 0
    for k in range(T):
        #if t+dt > T:
        #    dt = T-t
        t += dt
        j += 1
        h = boundary(h)  
        f = flux(h, d)
        q = production(h)
        h[i] = h[i] - dt/dx*(f[i]-f[i-1]) + dt*q[i]
        dt = cfl*dx/max(abs(df(h)))
    print(np.sum(h)*dx)
    return h