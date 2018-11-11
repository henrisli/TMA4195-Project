
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
alpha_s = 2*np.pi/180
Theta = rho*g*H*np.sin(alpha)
Theta_s = rho*g*H*np.sin(alpha_s)
kappa = 2*H**2/(Q*L)*mu*Theta**m
kappa_s = 2*H**2/(Q*L)*mu*Theta_s**m

def flux(h,d,dx):
    return kappa*np.power(h-d,m+2)/(m+2)

def shallowFlux(h,d,dx):
    h_x = np.append(0,np.diff(h))/dx
    h_x[1] = 0
    h_x[-1] = 0
    return kappa_s/(m+2)*np.power(np.abs(H/(L*np.tan(alpha_s))*h_x-1),m-1)*(1-H/(L*np.tan(alpha_s))*h_x)*np.power(h-d,m+2)

#from analytical import analytical

# The following imports a function for the boundary conditions
def inflow(h, n=0):
    if n == 0:
        h[0] = 0
    else:
        h[0:n] = 0
    return h

# The following computes the production q, given a height profile h
def production(h,*args):
    n = len(h) - 2
    q = np.zeros(n + 2)
    for i in range(n + 2):
        if i < n/3 + 1:
            q[i] = 1
        else:
            q[i] = 1-(i-(n/3+1))/(n/6)   
            
        if h[i]==0 and q[i]<0:
            q[i] = 0
    return q*1e+03

def retreating_production(h,k):
    n = len(h) - 2
    q = np.zeros(n + 2)
    for i in range(n + 2):
        if i < n/3 + 1 - k//100:
            q[i] = 1
        else:
            q[i] = 1-(i-(n/3 + 1 - k//100))/(n/6) 
            
        if h[i]==0 and q[i]<0:
            q[i] = 0
    return q*1e+03

# Solution of equation for height of glacier, both with classical
# and Godunov schemes..
def h_solution(method, T1, T2):
    # Solutions on coarser grids
    N  = 150
    dx = L/N
    
    #d = np.sin(np.linspace(-np.pi,np.pi,N+2))*6
    d = np.zeros(N+2)
    
    #Here we compute the maximum value of f'(u).
    s = np.linspace(0,50,1001)
    #dfv = max(np.diff(flux(s,np.sin(np.linspace(-np.pi,np.pi,1001))*6))/np.diff(s))
    dfv = max(np.diff(flux(s,np.zeros(1001),dx))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    
    if method == 'upw':
        # Coarser grid
        x  = np.arange(-0.5*dx,L+1.5*dx,dx)
        h0 = np.ones(N//3 + 1)*H
        #h0 = np.zeros(N//3 + 1)*H
        h0 = np.append(h0,np.zeros(N//3*2 + 1)*H)

        
        
        # Compute solutions with the three classical schemes
        hu = upw(h0, 0.995, dx, T1, flux, df, inflow, production, d)
        #hu_r = upw(hu, 0.995, dx, T2, flux, df, inflow, retreating_production, d)
        hu2 = upw(h0,0.995, dx, T1*10, flux, df, inflow, production, d)
        hu3 = upw(h0,0.995, dx, T1*100, flux, df, inflow, production, d)
        hu4 = upw(h0,0.995, dx, T1*1000, flux, df, inflow, production, d)
        hu5 = upw(h0,0.995, dx, T1*10000, flux, df, inflow, production, d)

        # Plot results
        plt.figure()
        plt.plot(x[1:-1], hu[1:-1], '-', markersize = 3) # We dont want to plot fictitious nodes, thereby the command [1:-1].
        #plt.plot(x[1:-1], hu_r[1:-1], '-', markersize = 3)
        plt.plot(x[1:-1], hu2[1:-1], '-', markersize = 3)
        plt.plot(x[1:-1], hu3[1:-1], '-', markersize = 3)
        plt.plot(x[1:-1], hu4[1:-1], '-', markersize = 3)
        plt.plot(x[1:-1], hu5[1:-1], '-', markersize = 3)
        #plt.plot(x[1:-1], d[1:-1], '-', markersize = 3)

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
        
        ug, phi = god(h0, 0.495, dx, T1, flux, df, inflow, production,d)
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


h_solution('upw', 1,2500)


def h_solution_11(method, T1, T2):
    # Solutions on coarser grids
    N  = 150
    dx = L/N
    
    #d = np.sin(np.linspace(-np.pi,np.pi,N+2))*6
    d = np.zeros(N+2)
    
    #Here we compute the maximum value of f'(u).
    s = np.linspace(0,50,1001)
    #dfv = max(np.diff(flux(s,np.sin(np.linspace(-np.pi,np.pi,1001))*6))/np.diff(s))
    dfv = max(np.diff(shallowFlux(s,np.zeros(1001),dx))/np.diff(s))
    print(dfv)
    df = lambda u: np.zeros(len(u)) + dfv
    
    
    if method == 'upw':
        # Coarser grid
        x  = np.arange(-0.5*dx,L+1.5*dx,dx)
        #h0 = np.ones(N//3 + 1)*H
        h0 = np.zeros(N//3 + 1)*H
        h0 = np.append(h0,np.zeros(N//3*2 + 1)*H)

        
        
        # Compute solutions with the three classical schemes
        hu = upw(h0, 0.995, dx, T1, shallowFlux, df, inflow, production, d)
        #hu_r = upw(hu, 0.995, dx, T2, flux, df, inflow, retreating_production, d)
        #hu2 = upw(h0,0.995, dx, T1*10, shallowFlux, df, inflow, production, d)
        #hu3 = upw(h0,0.995, dx, T1*100, shallowFlux, df, inflow, production, d)
        #hu4 = upw(h0,0.995, dx, T1*1000, shallowFlux, df, inflow, production, d)
        #hu5 = upw(h0,0.995, dx, T1*10000, shallowFlux, df, inflow, production, d)

        # Plot results
        plt.figure()
        plt.plot(x[1:-1], hu[1:-1], '-', markersize = 3) # We dont want to plot fictitious nodes, thereby the command [1:-1].
        #plt.plot(x[1:-1], hu_r[1:-1], '-', markersize = 3)
        #plt.plot(x[1:-1], hu2[1:-1], '-', markersize = 3)
        #plt.plot(x[1:-1], hu3[1:-1], '-', markersize = 3)
        #plt.plot(x[1:-1], hu4[1:-1], '-', markersize = 3)
        #plt.plot(x[1:-1], hu5[1:-1], '-', markersize = 3)
        #plt.plot(x[1:-1], d[1:-1], '-', markersize = 3)

        plt.title("Upwind")
#h_solution_11("upw",100,0)


def film():
    s = np.linspace(0,50,1001)
    d = np.sin(np.linspace(-np.pi,np.pi,1001))*6
    dfv = max(np.diff(flux(s,d))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    
    # Solutions on coarser grids
    N  = 150
    dx = L/N
    d = np.sin(np.linspace(-np.pi,np.pi,N+2))*6
    d = np.zeros(N+2)
    
    #h0 = np.ones(N//3+1)*H
    h0 = np.zeros(N//3+1)
    h0 = np.append(h0,np.zeros(N//3*2+1))
    x = np.arange(-0.5*dx, L + 1.5*dx,dx)
    hu = upw2(h0,0.995, dx, 8000, 10000, flux, df, inflow, production, retreating_production, d) + d
    plt.figure()
    
    tvalues = np.arange(1000)
    fig = plt.figure()
    xvalues = x
    xg = xvalues
    yg = tvalues
    xg, yg = np.meshgrid(xg, yg)
    y1 = hu
    fig, ax = plt.subplots()
    
    line, = ax.plot(xvalues, np.linspace(-6,50,152))
    def animate(i):
        line.set_ydata(y1[i])
        return line,
    def init():
        line.set_ydata(np.ma.array(xvalues, mask=True))
        return line,
    
    ax.ani = animation.FuncAnimation(fig, animate, np.arange(1, 8000+10000+1), init_func = init,
                                  interval = 1, blit=True)
    
    plt.show()
#film()
