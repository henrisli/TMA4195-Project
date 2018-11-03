
import numpy as np
import matplotlib.pyplot as plt

# Import schemes:
from upw import upw

# Height equation flux function
H = 50
L = 1000
Q = 7.5/(365*24*3600)
mu = 9.3e-25
m = 3
rho = 1000
g = 9.81
alpha = 5*np.pi/180
Theta = rho*g*H*np.sin(alpha)
def flux(h):
    return 2*H**2/(Q*L)*mu*Theta**m*np.power(h,m+2)/(m+2)
#from analytical import analytical

# The following imports a function for the boundary conditions
from inflow import inflow


# Solution of equation for height of glacier, both with classical
# and Godunov schemes..
def h_solution(method, T):
    q = np.repeat(1,51)
    q = np.append(q,np.linspace(1,-1,50))
    q = np.append(q,np.repeat(0,51))*Q
    #Here we compute the maximum value of f'(u).
    s = np.linspace(0,1,1001)
    dfv = max(np.diff(flux(s))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
        
    # Solutions on coarser grids
    N  = 150
    dx = 1/N
    
    if method == 'classical':
        # Coarser grid
        x  = np.arange(-0.5*dx,1+1.5*dx,dx)
        h0 = np.ones(len(x))
        
        # Compute solutions with the three classical schemes
        hu = upw(h0, 0.995, dx, T, flux, df, inflow, q)
                
        
        # Plot results
        plt.figure()
        # Analytical solution:
        #plt.plot(xr, analytical(xr, T), color = 'red')
        plt.plot(x[1:-1], hu[1:-1], '.', markersize = 3) # We dont want to plot fictitious nodes, thereby the command [1:-1].
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