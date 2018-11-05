import numpy as np

def upw2(h0, cfl, dx, T, flux, df, boundary, production):
    a = [[0 for i in range(len(h0))] for i in range(T//10+1)]
    b = np.array(a)
    h = np.copy(h0)
    b[0] = h
    t = 0.0
    dt = cfl*dx/max(abs(df(h0)))
    print(dt)
    print(dx)
    i = np.arange(1,len(h0)-1,1)
    j = 0
    for k in range(T):
        #if t+dt > T:
        #    dt = T-t
        t += dt
        j += 1
        h = boundary(h)  
        f = flux(h)
        q = production(h)
        #print(f)
        h[i] = h[i] - dt/dx*(f[i]-f[i-1]) + dt*q[i]
        #print(h)
        dt = cfl*dx/max(abs(df(h)))
        if (k+1)%10 == 0:
            b[(k+1)//10] = h
    return b