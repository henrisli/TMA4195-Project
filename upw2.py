import numpy as np

def upw2(h0, cfl, dx, T1, T2, flux, df, boundary, production1, production2):
    b = np.array([[0.0 for i in range(len(h0)-2)] for i in range(T1+T2+1)])
    h = np.copy(h0)
    b[0] = h[1:-1]
    dt = cfl*dx/max(abs(df(h0)))
    i = np.arange(1,len(h0)-1,1)
    # Advancing glacier
    for k in range(T1):
        h = boundary(h)  
        f = flux(h, dx)
        q = production1(h,k)
        h[i] = h[i] - dt/dx*(f[i]-f[i-1]) + dt*q[i]
        h[h<1e-16] = 0
        dt = cfl*dx/max(abs(df(h)))
        b[k+1] = h[1:-1]
    # Retreating glacier
    for k in range(T2):
        h = boundary(h)  
        f = flux(h, dx)
        q = production2(h,k)
        h[i] = h[i] - dt/dx*(f[i]-f[i-1]) + dt*q[i]
        h[h<1e-16] = 0
        dt = cfl*dx/max(abs(df(h)))
        b[(T1+k+1)] = h[1:-1]
    return b