
import numpy as np


def upw(h0, cfl, dx, T, flux, df, boundary,q):
    h = np.copy(h0)
    t = 0.0
    dt = cfl*dx/max(abs(df(h0)))
    print(dt)
    print(dx)
    i = np.arange(1,len(h0)-1,1)
    j = 1
    while t<T:
        if t+dt > T:
            dt = T-t
        t += dt
        j += 1
        h = boundary(h)  
        f = flux(h)
        print(f)
        h[i] = h[i] - dt/dx*(f[i]-f[i-1]) + dt*q[i]
        print(h)
        dt = cfl*dx/max(abs(df(h)))
    print(j)
    return h