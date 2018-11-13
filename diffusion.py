
import numpy as np

def diffusion(dx,K,Dright,Dleft,h0,tf,q,d,gamma,k):
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
        q_p = q(H,k)
        H[1:-1] = H[1:-1] + mu_x*Dright*(gamma*(Hb[2:] - Hb[1:-1])-dx) - mu_x*Dleft*(gamma*(Hb[1:-1]-Hb[:-2])-dx)+ dt*q_p[1:-1]
        H[H<1e-06] = 0
        t = t + dt
        count = count + 1
    dtav = tf / count
    #print(count)
    return H, dtav, q_p[1:-1]
