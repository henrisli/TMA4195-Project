import numpy as np

def explicit_scheme(dx,K,H0,dt,tf,production,d,boundary, gamma, Gamma, m):
    H = np.copy(H0)
    t = 0.0
    j = 0
    while t < tf:
        if t+dt > tf:
            dt = tf-t
        H = boundary(H)
        Hrt = 0.5*(H[2:] + H[1:-1])
        Hlt = 0.5*(H[1:-1] + H[:-2])
        a2rt = np.power(np.abs(gamma*(np.diff(H)[1:])/dx-1),m-1)
        a2lt = np.power(np.abs(gamma*(np.diff(H)[:-1])/dx-1),m-1)
        Drt = Gamma * np.power(Hrt,m+2) * a2rt
        Dlt = Gamma * np.power(Hlt,m+2) * a2lt
        mu_x = dt/(dx*dx)
        Hb = H + d
        q_p = production(H,j)
        H[1:-1] = H[1:-1] + mu_x*Drt*(gamma*(Hb[2:] - Hb[1:-1])-dx) - mu_x*Dlt*(gamma*(Hb[1:-1]-Hb[:-2])-dx)+ dt*q_p[1:-1]
        H[H<1e-06] = 0
        t += dt
        j += 1
    print(j)
    return H