import numpy as np

def explicit_scheme2(dx,K,H0,dt,tf1,tf2,production1,production2,d,boundary, gamma, Gamma, m):
    b = np.array([[0.0 for i in range(len(H0)-2)] for i in range(tf1+tf2+1)])
    H = np.copy(H0)
    b[0] = H[1:-1]
    t = 0.0
    for j in range(tf1):
#        if t+dt > tf1:
#            dt = tf1-t
        H = boundary(H)
        Hrt = 0.5*(H[2:] + H[1:-1])
        Hlt = 0.5*(H[1:-1] + H[:-2])
        a2rt = np.power(np.abs(gamma*(np.diff(H)[1:])/dx-1),m-1)
        a2lt = np.power(np.abs(gamma*(np.diff(H)[:-1])/dx-1),m-1)
        Drt = Gamma * np.power(Hrt,m+2) * a2rt
        Dlt = Gamma * np.power(Hlt,m+2) * a2lt
        mu_x = dt/(dx*dx)
        Hb = H + d
        q_p = production1(H,j)
        H[1:-1] = H[1:-1] + mu_x*Drt*(gamma*(Hb[2:] - Hb[1:-1])-dx) - mu_x*Dlt*(gamma*(Hb[1:-1]-Hb[:-2])-dx)+ dt*q_p[1:-1]
        H[H<1e-06] = 0
        t += dt
        b[j+1] = H[1:-1]
#    t = 0.0
    for k in range(tf2):
#        if t+dt > tf2:
#            dt = tf2-t
        H = boundary(H)
        Hrt = 0.5*(H[2:] + H[1:-1])
        Hlt = 0.5*(H[1:-1] + H[:-2])
        a2rt = np.power(np.abs(gamma*(np.diff(H)[1:])/dx-1),m-1)
        a2lt = np.power(np.abs(gamma*(np.diff(H)[:-1])/dx-1),m-1)
        Drt = Gamma * np.power(Hrt,m+2) * a2rt
        Dlt = Gamma * np.power(Hlt,m+2) * a2lt
        mu_x = dt/(dx*dx)
        Hb = H + d
        q_p = production2(H,k)
        H[1:-1] = H[1:-1] + mu_x*Drt*(gamma*(Hb[2:] - Hb[1:-1])-dx) - mu_x*Dlt*(gamma*(Hb[1:-1]-Hb[:-2])-dx)+ dt*q_p[1:-1]
#        if np.max(H) > 0.5: print(np.max(H))
        H[H<1e-06] = 0
        t += dt
        b[(tf1+k+1)] = H[1:-1]
    return b