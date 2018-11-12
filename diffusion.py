
import numpy as np

def diffusion(Lx,K,Dright,Dleft,h0,tf,q,d):
    dx = 2 * Lx / K
    x  = np.arange(-0.5*dx,Lx+1.5*dx,dx)
    t = 0.0
    H = np.copy(h0)
    count = 0
    while t < tf:
        maxD = [max(Dleft), max(Dright)]
        maxD = max(maxD)
        if maxD <= 0.0:
            dt = tf - t
        else:
            dt0 = 0.495 * dx**2 / maxD;
            dt = min(dt0, tf - t)
        
        mu_x = dt / (dx*dx)
        Hb = H + d
        H[1:-1] = H[1:-1] + mu_x*Dright*(Hb[2:] - Hb[1:-1]) - mu_x*Dleft*(Hb[1:-1]-Hb[:-2])
        H[H<1e-06] = 0
        q_p = q(H)
        H[1:-1] = H[1:-1] + q_p[1:-1] * dt
        t = t + dt
        count = count + 1
    dtav = tf / count
    return H, dtav
