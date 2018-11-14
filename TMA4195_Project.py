
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D


# Import schemes:
from upw import upw
from upw2 import upw2
#from god import god
from explicit_scheme import explicit_scheme
from steady_state import StationaryGlacier

# Height equation flux function
H = 100
H0 = 0.5
L = 6000
Q = 2/(365*24*3600)
mu = 9.3e-25
m = 3
rho = 1000
g = 9.81
alpha = 20*np.pi/180
alpha_s = 3*np.pi/180
Theta = rho*g*H*np.sin(alpha)
Theta_s = rho*g*H*np.sin(alpha_s)
kappa = 2*H**2/(Q*L)*mu*Theta**m
kappa_s = 2*H**2/(Q*L)*mu*Theta_s**m
Gamma = kappa_s/(m+2)
gamma = H/(L*np.tan(alpha_s))

def flux(h, dx):
    return kappa*np.power(h,m+2)/(m+2)

def shallowFlux(h, dx):
    h_x = np.append(0,np.diff(h))/dx
    h_x[1] = 0
    h_x[-1] = 0
    return kappa_s/(m+2)*np.power(np.abs(1-gamma*h_x),m-1)*(1-gamma*h_x)*np.power(h,m+2)


#from analytical import analytical

# The following imports a function for the boundary conditions
def inflow(h):
    h[0] = H0
    return h

# The following computes the production q, given a height profile h
def production(h,*args):
    n = len(h) - 2
    q = np.zeros(n + 2)
    for i in range(n + 2):
        if i < n/3:
            q[i] = 1
        else:
            q[i] = 1-(i-(n/3))/(n/12)   
            
        if h[i]<1e-15 and q[i]<1e-16:
            q[i] = 0
    return q

def retreating_production(h,k):
    n = len(h) - 2
    q = np.zeros(n + 2)
    for i in range(n + 2):
        if i < n/3 - k//590:
            q[i] = 1
        else:
            q[i] = 1-(i-(n/3 - k//590))/(n/12) 
            
        if h[i]<1e-16 and q[i]<1e-16:
            q[i] = 0
    return q

def advancing_production(h,k):
    n = len(h) - 2
    k -= 8261
    if k//590>n/3:
        k = 590*(n/3)
    q = np.zeros(n+2)
    for i in range(n+2):
        if i < k//590:
            q[i] = 1
        else:
            q[i] = 1 - (i - k//590)/(n/12)
            
        if h[i]<1e-16 and q[i] < 1e-16:
            q[i] = 0
    # Full production after 43661 iterations or 409 years
    return q


def retreating_shallow_production(h,k):
    n = len(h) - 2
    q = np.zeros(n + 2)
    for i in range(n + 2):
        if i < n/3 - k//670:
            q[i] = 1
        else:
            q[i] = 1-(i-(n/3 - k//670))/(n/12) 
            
        if h[i]<1e-16 and q[i]<1e-16:
            q[i] = 0
    return q

def advancing_shallow_production(h,k):
    n = len(h) - 2
    k -= 8041
    if k//670>n/3:
        k = 670*(n/3)
    q = np.zeros(n+2)
    for i in range(n+2):
        if i < k//670:
            q[i] = 1
        else:
            q[i] = 1 - (i - k//670)/(n/12)
            
        if h[i]<1e-16 and q[i] < 1e-16:
            q[i] = 0
    # Full production after 41541 iterations or 412 years
    return q

# Solution of equation for height of glacier, both with classical
# and Godunov schemes..
def h_solution(method, T1, T2, T3, T4, T5, production, mov):
    # Solutions on coarser grids
    N  = 180
    dx = 1/N
    
    #Here we compute the maximum value of f'(u).
    s = np.linspace(0,2,1001)
    dfv = max(np.diff(flux(s,dx))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    
    if method == 'upw':
        # Coarser grid
        x  = np.arange(-0.5*dx,1+1.5*dx,dx)
        G = StationaryGlacier(H, H0, L, Q*(365*24*3600), mu, m, rho, g, alpha*180/np.pi, 1/3 ,2/3)
        G.generateLinearQ()
        if mov == "advancing":
            h0 = np.zeros(N+2)
        elif mov == "retreating":
            h0 = G.getHeight(x)
        
        # Compute solutions with the three classical schemes
        hu1, t1 = upw(h0, 0.995, dx, T1, flux, df, inflow, production)
        hu2, t2 = upw(h0,0.995, dx, T2, flux, df, inflow, production)
        hu3, t3 = upw(h0,0.995, dx, T3, flux, df, inflow, production)
        hu4, t4 = upw(h0,0.995, dx, T4, flux, df, inflow, production)
        hu5, t5 = upw(h0,0.995, dx, T5, flux, df, inflow, production)
        
        
        # Plot results
        plt.figure()
        plt.plot(x[1:-1]*L, hu1[1:-1]*H, '-', markersize = 3, label = str(round(T1*50)) + " years") # We dont want to plot fictitious nodes, thereby the command [1:-1].
        plt.plot(x[1:-1]*L, hu2[1:-1]*H, '-', markersize = 3, label = str(round(T2*50)) + " years")
        plt.plot(x[1:-1]*L, hu3[1:-1]*H, '-', markersize = 3, label = str(round(T3*50)) + " years")
        plt.plot(x[1:-1]*L, hu4[1:-1]*H, '-', markersize = 3, label = str(round(T4*50)) + " years")
        plt.plot(x[1:-1]*L, hu5[1:-1]*H, '-', markersize = 3, label = str(round(T5*50)) + " years")
        
        plt.plot(x[1:-1]*L, G.getHeight(x[1:-1])*H, '-', markersize = 3, label = "Steady State")

        plt.legend(loc = 1, fontsize = 7)
        plt.xlabel("Length (m)")
        plt.ylabel("Height (m)")
        if mov == "advancing":
            plt.title("Height profile of advancing glacier")
            plt.savefig("Advancing_glacier.pdf")
        elif mov == "retreating":
            plt.title("Height profile of retreating glacier")
            plt.savefig("Retreating_glacier.pdf")
        
        """
    
    elif method == 'god':
        # Coarser grid, need two fictitious nodes at each end for this scheme.
        xh = np.arange(-1.5*dx, L + 2.5*dx, dx)
        #h0 = np.ones(N//3 + 2)*H
        h0 = np.zeros(N//3 + 2)*H
        h0 = np.append(h0,np.zeros(N//3*2 + 2))
        
        ug, phi, t = god(h0, 0.495, dx, T1, flux, df, inflow, production,d)
        print(t)
        #Plot results
        plt.figure()
        plt.plot(xh[2:-2], ug[2:-2], '-', markersize = 3)
        plt.title("Godunov")
        # The following commented out section saves the plots
        
        """
#Advancing:
#h_solution('upw', 2, 4, 6, 8.18, 10, advancing_production, "advancing")

#Retreating
#h_solution('upw', 2, 4, 6, 8.18, 13, retreating_production, "retreating")
plt.plot(np.arange(182)*6000/180, advancing_production(np.ones(182),10000000000))
plt.plot(np.arange(182)*6000/180,np.zeros(182))

def h_solution_11(T1,T2,T3,T4,T5, production, mov):
    # Solutions on coarser grids
    N  = 150
    dx = 1/N
    
    #d = np.sin(np.linspace(-np.pi,np.pi,N+2))*6
    d = np.zeros(N+2)
    
    #Here we compute the maximum value of f'(u).
    s = np.linspace(0,0.9,1001)
    dfv = max(np.diff(shallowFlux(s,dx))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv

    
    # Coarser grid
    x  = np.arange(-0.5*dx,1+1.5*dx,dx)
    h0 = np.zeros(N + 2)
    dt = 0.495*dx*dx/max(abs(df(h0)))
    print("dt: ", dt)
    if mov == "retreating":
        h0 = explicit_scheme(dx,N,h0,dt,12,advancing_shallow_production,d,inflow, gamma, Gamma, m)


    # Compute solutions with the three classical schemes
    hu1 = explicit_scheme(dx, N, h0, dt, T1, production, d, inflow, gamma, Gamma, m)
    hu2 = explicit_scheme(dx, N, h0, dt, T2, production, d, inflow, gamma, Gamma, m)
    hu3 = explicit_scheme(dx, N, h0, dt, T3, production, d, inflow, gamma, Gamma, m)
    hu4 = explicit_scheme(dx, N, h0, dt, T4, production, d, inflow, gamma, Gamma, m)
    hu5 = explicit_scheme(dx, N, h0, dt, T5, production, d, inflow, gamma, Gamma, m)

    # Plot results
    plt.figure()
    plt.plot(x[1:-1]*L, hu1[1:-1]*H, '-', markersize = 3, label = str(round(T1*50)) + " years")
    plt.plot(x[1:-1]*L, hu2[1:-1]*H, '-', markersize = 3, label = str(round(T2*50)) + " years")
    plt.plot(x[1:-1]*L, hu3[1:-1]*H, '-', markersize = 3, label = str(round(T3*50)) + " years")
    plt.plot(x[1:-1]*L, hu4[1:-1]*H, '-', markersize = 3, label = str(round(T4*50)) + " years")
    if mov=="retreating":
          plt.plot(x[1:-1]*L, h0[1:-1]*H, '-', markersize = 3, label = "0 years") # We dont want to plot fictitious nodes, thereby the command [1:-1].  
    plt.plot(x[1:-1]*L, hu5[1:-1]*H, '-', markersize = 3, label = str(round(T5*50)) + " years")
    plt.xlabel("Length (m)")
    plt.ylabel("Height (m)")
    
    plt.legend(loc = 1, fontsize = 7)
    if mov=="advancing":
        plt.title("Height profile of advancing gentle slope glacier")
        plt.savefig("Advancing_glacier_gentle.pdf")
    elif mov=="retreating":
        plt.title("Height profile of retreating gentle slope glacier")
        plt.savefig("Retreating_glacier_gentle.pdf")
        
#Advancing glacier:
#h_solution_11(2,4,6,8.24,12, advancing_shallow_production, "advancing")

#Retreating glacier:
#h_solution_11(2,4,6,8.24,16, retreating_shallow_production, "retreating")



def film(T1,T2):    
    # Solutions on coarser grids
    N  = 180
    dx = 1/N
    
    s = np.linspace(0,2,1001)
    dfv = max(np.diff(flux(s,dx))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    dt = 0.995 * dx / dfv
    
    #h0 = np.ones(N//3+1)*H
    h0 = np.zeros(N//3+1)
    h0 = np.append(h0,np.zeros(N//3*2+1))
    x = np.arange(-0.5*dx, 1 + 1.5*dx,dx)
    hu = upw2(h0,0.995, dx, T1, T2, flux, df, inflow, advancing_production, retreating_production)
    plt.figure()
    
    
    tvalues = np.arange(200)
    fig = plt.figure()
    xvalues = x[1:-1]*L
    xg = xvalues
    yg = tvalues
    xg, yg = np.meshgrid(xg, yg)
    y1 = hu*H
    
    G = StationaryGlacier(H, H0, L, Q*(365*24*3600), mu, m, rho, g, alpha*(180/np.pi), 1/3 ,2/3)
    G.generateLinearQ()
    
    fig, ax = plt.subplots()
#    ax.plot(xvalues, G.getHeight(x[1:-1])*H, color='tab:orange')
    
    iter_per_frame = 1000
#    time_steps = 10
    
    line, = ax.plot(xvalues, np.linspace(-6,H,N), color='tab:blue')
    line2, = ax.plot(xvalues, np.linspace(-6,H,N), color='tab:orange')
    fill = ax.fill_between(xvalues, 0, np.linspace(-6,H,N), color='tab:blue', interpolate = True)
    antifill = ax.fill_between(xvalues, np.linspace(-6,H,N), H, color='white', interpolate = True)
    text = ax.text(0.8*L, 0.5*H, '')
    def animate(i):
        line.set_ydata(y1[i])
        
#        Fill between and let yellow be on top 
        if i < T1:
            q_vector = advancing_production(np.ones(N), i)
            q_func = lambda z: np.interp(z, np.linspace(0, 1, N), q_vector)
            G.setQ(q_func)
            G.calculateHeight()
            line2.set_ydata(G.getHeight(x[1:-1])*H)
        else:
            q_vector = retreating_production(np.ones(N), i-T1)
            q_func = lambda z: np.interp(z, np.linspace(0, 1, N), q_vector)
            G.setQ(q_func)
            G.calculateHeight()
            line2.set_ydata(G.getHeight(x[1:-1])*H)
            
        fill = ax.fill_between(xvalues, 0, y1[i], color='tab:blue', interpolate = True)
        antifill = ax.fill_between(xvalues, y1[i], 1.05*H, color='white', interpolate = True)
        
        
#        line2.set_ydata(
#        ax.fill_between(xvalues, np.zeros(np.shape(y1[0])), y1[i], interpolate=True, facecolor='tab:blue')
#        ax.fill_between(xvalues, G.getHeight(x[1:-1])*H, y1[i], interpolate=True, facecolor='white')
        text.set_text('T = {:.0f} years'.format(i*dt*100))
        return [antifill, fill, line, line2, text]

    def init():
        line.set_ydata(np.ma.array(xvalues, mask=True))
        line2.set_ydata(np.ma.array(xvalues, mask=True))
        return [antifill, fill, line, line2, text]
    
    ax.legend(['$h(x, t)$', '$h_0(x)$'], loc = 1)
    
    ax.ani = animation.FuncAnimation(fig, animate, np.arange(1, T1+T2+1, iter_per_frame), init_func = init,interval = 1, blit=True)
    
    ax.ani.save('file.gif', fps = 60, writer = 'imagemagick')
#    ax.ani.to_html5_video(embed_limit=None)
    
#    plt.show()
    
#film(37000,37000)



def Plot_3D(T1,T2):    
    # Solutions on coarser grids
    N  = 180
    dx = 1/N
    
    s = np.linspace(0,2,1001)
    dfv = max(np.diff(flux(s,dx))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    G = StationaryGlacier(H, H0, L, Q*(365*24*3600), mu, m, rho, g, alpha*180/np.pi, 1/3 ,2/3)
    G.generateLinearQ()

    
    #h0 = np.ones(N//3+1)*H
    h0 = np.zeros(N//3+1)
    h0 = np.append(h0,np.zeros(N//3*2+1))
    x = np.arange(-0.5*dx, 1 + 1.5*dx,dx)
    hu1 = upw2(h0,0.995, dx, T1, 0, flux, df, inflow, advancing_production, retreating_production)*H
    #hu2 = upw2(G.getHeight(x),0.995, dx, 0, T2, flux, df, inflow, advancing_production, retreating_production)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    #x = np.arange(N+2)*dx*L
    #y = np.arange(T1)
    x = np.arange(T1+1)
    y = np.arange(N)
    x, y = np.meshgrid(x, y)
    #z = np.array( [ np.sin( 1e-3 * t * np.arange(100)) for t in x ] )
    z = hu1.transpose()
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='Blues')
    plt.show()
    
    
def steady_state_comparison(angle):  
    # Solutions on coarser grids
    N  = 150
    dx = 1/N
    
    #d = np.sin(np.linspace(-np.pi,np.pi,N+2))*6
    d = np.zeros(N+2)
    
    #Here we compute the maximum value of f'(u).
    s = np.linspace(0,0.9,1001)
    dfv = max(np.diff(shallowFlux(s,dx))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    alpha_u = alpha
    if angle == "gentle":
        alpha_u = alpha_s
    
    # Coarser grid
    x  = np.arange(-0.5*dx,1+1.5*dx,dx)
    h0 = np.zeros(N + 2)
    dt = 0.495*dx*dx/max(abs(df(h0)))
    print("dt: ", dt)
    G = StationaryGlacier(H, H0, L, Q*(365*24*3600), mu, m, rho, g, alpha_u*180/np.pi, 1/3 ,2/3)
    G.generateLinearQ()
    Theta_u = rho*g*H*np.sin(alpha_u)
    kappa_u = 2*H**2/(Q*L)*mu*Theta_u**m
    Gamma = kappa_u/(m+2)
    gamma = H/(L*np.tan(alpha_u))
    
    # Compute solutions with the three classical schemes
    hu1 = explicit_scheme(dx, N, h0, dt, 10, production, d, inflow, gamma, Gamma, m)
    
    # Plot results
    plt.figure()
    plt.plot(x[1:-1]*L, hu1[1:-1]*H, '-', markersize = 3, label = "Expanded model")
    plt.plot(x[1:-1]*L, G.getHeight(x[1:-1])*H, '-', markersize = 3, label = "Simplified model")
    plt.xlabel("Length (m)")
    plt.ylabel("Height (m)")
    
    plt.legend(loc = 1, fontsize = 7)
    plt.title("Comparison of steady state solutions of height profile of " + angle + " glacier", fontsize = 9)
    plt.savefig("Steady_state_comparison_" + angle + ".pdf")
#steady_state_comparison("steep")