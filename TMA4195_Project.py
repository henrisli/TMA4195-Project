
import numpy as np
import matplotlib.pyplot as plt

# Import schemes:
from upw import upw
from god import god

# Height equation flux function
H = 50
L = 2000
Q = 3/(365*24*3600)
mu = 9.3e-25
m = 3
rho = 1000
g = 9.81
alpha = 45*np.pi/180
Theta = rho*g*H*np.sin(alpha)
def flux(h):
    return 2*H**2/(Q*L)*mu*Theta**m*np.power(h,m+2)/(m+2)

#from analytical import analytical

# The following imports a function for the boundary conditions
def inflow(h, n=0):
    if n == 0:
        h[0] = 0
    else:
        h[0:n] = 0
    return h

# The following computes the production q, given a height profile h
def production(h):
    n = len(h) - 2
    q = np.zeros(n + 2)
    for i in range(n + 2):
        if h[i]>0:
            if i < n/3 + 1:
                q[i] = 1
            #elif i < 2*n/3 + 2:
            #    q[i] = 1-(i-(n/3+1))/(n/6)
            else:
                q[i] = 1-(i-(n/3+1))/(n/6)    
    
    return q*1e+03

# Solution of equation for height of glacier, both with classical
# and Godunov schemes..
def h_solution(method, T):
    #q = np.append(q,np.linspace(1,-1,101))
    #q = np.append(q,np.repeat(0,51))*Q
    #Here we compute the maximum value of f'(u).
    s = np.linspace(0,50,1001)
    dfv = max(np.diff(flux(s))/np.diff(s))
    print(dfv)
    df = lambda u: np.zeros(len(u)) + dfv
    
        
    # Solutions on coarser grids
    N  = 150
    dx = L/N
    
    if method == 'classical':
        # Coarser grid
        x  = np.arange(-0.5*dx,L+1.5*dx,dx)
        #h0 = np.ones(len(x))*H
        h0 = np.ones(51)*H
        h0 = np.append(h0,np.zeros(101))
        
        # Compute solutions with the three classical schemes
        hu = upw(h0, 0.995, dx, T, flux, df, inflow, production)
        hu2 = upw(h0,0.995, dx, T*10, flux, df, inflow, production)
        # Plot results
        plt.figure()
        # Analytical solution:
        #plt.plot(xr, analytical(xr, T), color = 'red')
        plt.plot(x[1:-1], hu[1:-1], '-', markersize = 3) # We dont want to plot fictitious nodes, thereby the command [1:-1].
        plt.plot(x[1:-1], hu2[1:-1], '-', markersize = 3)
        #plt.plot(x[1:-1], np.repeat(100,len(hu[1:-1])))
        plt.title("Upwind")
        # The following commented out section saves the plots
        """
        if T == 1:
            plt.savefig("solution_classical_cont.pdf")
        elif T == 0.5:
            plt.savefig("solution_classical_discont.pdf")
        """
    """
    elif method == 'high':
        # Coarser grid, need two fictitious nodes at each end for this scheme.
        xh = np.arange(-1.5*dx, 1 + 2.5*dx, dx)
        
        # Discontinuous solution:
        if T == 0.5:
            u0 = np.zeros(len(xh))
            u0[xh<=0] = 1.0
            
        # Continuous initial:
        elif T == 1:
            u0 = np.ones(len(xh))*analytical(1,T)
            u0[xh<=0] = 1.0
        
        ug, phi = god(u0, 0.495, dx, T, flux, df, inflow)
        
        #Plot results
        plt.figure()
        # Analytical solution:
        plt.plot(xr, analytical(xr, T), color = 'red')
        plt.plot(xh[2:-2], ug[2:-2], '.', markersize = 3)
        plt.title("Godunov")
        # The following commented out section saves the plots
        
        if T == 0.5:
            plt.savefig("solution_high_discont.pdf")
        elif T == 1:
            plt.savefig("solution_high_cont.pdf")
        """


h_solution('classical', 10000)