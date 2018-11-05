
import numpy as np
import matplotlib.pyplot as plt

# Import schemes:
from upw import upw
from upw2 import upw2
from god import god

# Height equation flux function
H = 50
L = 2000
Q = 3.5/(365*24*3600)
mu = 9.3e-25
m = 3
rho = 1000
g = 9.81
alpha = 25*np.pi/180
Theta = rho*g*H*np.sin(alpha)
kappa = 2*H**2/(Q*L)*mu*Theta**m

def flux(h,d):
    return kappa*np.power(h-d,m+2)/(m+2)

def shallowFlux(h):
    return kappa/(m+2)*np.power(np.abs(H/(L*np.tan(alpha))*np.diff(h)-1),m-1)*(1-H/(L*np.tan(alpha))*np.diff(h))*np.power(h,m+2)

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
            else:
                q[i] = 1-(i-(n/3+1))/(n/6)    
    
    return q*1e+03

# Solution of equation for height of glacier, both with classical
# and Godunov schemes..
def h_solution(method, T):
    # Solutions on coarser grids
    N  = 150
    dx = L/N
    
    #d = np.sin(np.linspace(-np.pi,np.pi,N+2))*6
    d = np.zeros(N+2)
    
    #Here we compute the maximum value of f'(u).
    s = np.linspace(0,50,1001)
    #dfv = max(np.diff(flux(s,np.sin(np.linspace(-np.pi,np.pi,1001))*6))/np.diff(s))
    dfv = max(np.diff(flux(s,np.zeros(1001)))/np.diff(s))
    print(dfv)
    df = lambda u: np.zeros(len(u)) + dfv
    
    
    if method == 'upw':
        # Coarser grid
        x  = np.arange(-0.5*dx,L+1.5*dx,dx)
        #h0 = np.ones(len(x))*H
        h0 = np.ones(N//3 + 1)*H
        h0 = np.append(h0,np.zeros(N//3*2 + 1))
        
        
        # Compute solutions with the three classical schemes
        hu = upw(h0, 0.995, dx, T, flux, df, inflow, production, d)
        hu2 = upw(h0,0.995, dx, T*10, flux, df, inflow, production, d)
        hu3 = upw(h0,0.995, dx, T*100, flux, df, inflow, production, d)
        hu4 = upw(h0,0.995, dx, T*1000, flux, df, inflow, production, d)
        hu5 = upw(h0,0.995, dx, T*10000, flux, df, inflow, production, d)

        # Plot results
        plt.figure()
        plt.plot(x[1:-1], hu[1:-1], '-', markersize = 3) # We dont want to plot fictitious nodes, thereby the command [1:-1].
        plt.plot(x[1:-1], hu2[1:-1], '-', markersize = 3)
        plt.plot(x[1:-1], hu3[1:-1], '-', markersize = 3)
        plt.plot(x[1:-1], hu4[1:-1], '-', markersize = 3)
        plt.plot(x[1:-1], hu5[1:-1], '-', markersize = 3)
        plt.plot(x[1:-1], d[1:-1], '-', markersize = 3)

        plt.title("Upwind")
        # The following commented out section saves the plots
        """
        if T == 1:
            plt.savefig("solution_classical_cont.pdf")
        elif T == 0.5:
            plt.savefig("solution_classical_discont.pdf")
        """
    
    elif method == 'god':
        # Coarser grid, need two fictitious nodes at each end for this scheme.
        xh = np.arange(-1.5*dx, L + 2.5*dx, dx)
        #h0 = np.ones(len(x))*H
        h0 = np.ones(N//3 + 2)*H
        h0 = np.append(h0,np.zeros(N//3*2 + 2))
        
        ug, phi = god(h0, 0.495, dx, T, flux, df, inflow, production)
        #Plot results
        plt.figure()
        plt.plot(xh[2:-2], ug[2:-2], '-', markersize = 3)
        plt.title("Godunov")
        # The following commented out section saves the plots
        """
        if T == 0.5:
            plt.savefig("solution_high_discont.pdf")
        elif T == 1:
            plt.savefig("solution_high_cont.pdf")
        """


h_solution('upw', 1)






def film():
    s = np.linspace(0,50,1001)
    dfv = max(np.diff(flux(s))/np.diff(s))
    print(dfv)
    df = lambda u: np.zeros(len(u)) + dfv
    
    
    # Solutions on coarser grids
    N  = 150
    dx = L/N
    
    
    h0 = np.ones(51)*H
    h0 = np.append(h0,np.zeros(101))
    x = np.arange(-0.5*dx, L + 1.5*dx,dx)
    hu = upw2(h0,0.995, dx, 10000, flux, df, inflow, production, d)
    print(hu)
    plt.figure()
    for i in hu:
        plt.plot(x[1:-1], i[1:-1], '-', markersize = 3) # We dont want to plot fictitious nodes, thereby the command [1:-1].
    
    from matplotlib import animation
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2000), ylim=(-1, 51))
    line, = ax.plot([], [], lw=2)
    
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,
    
    # animation function.  This is called sequentially
    def animate(i):
        x = np.arange(-0.5*dx, L + 1.5*dx,dx)
        y = hu[i]
        line.set_data(x, y)
        return line,
    
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=1000, interval=20, blit=True)
    
    plt.show()
#film()