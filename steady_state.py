
import numpy as np
import matplotlib.pyplot as plt

class StationaryGlacier:
    """Compute a stationary glacier for a given set of constants. 

    Args:
        H (float): Scale of height
        H_0 (float): Height at x = 0, input as a multiple of H
        L (float): Scale of length
        Q (float): Downpour in meters per year
        mu (float): Glacier modelling parameter mu
        m (float):  Glacier modelling parameter m
        density (float): Density of modelled fluid
        angle (float): Slope of glacier in degrees
        x_s (float): Start, required to use setQ(...)
        x_f (float): End, required to use setQ(...)
    """
    
    def __init__(self, heightScale, heightStart, lengthScale, downpourYear, 
                 parameterMu, parameterM, density, gravity, angleInDegrees, x_s, x_f):
        
        
        

        self.H = heightScale
        self.h_0 = heightStart
        self.L = lengthScale
        self.Q = downpourYear / (24 * 365 * 3600)
        self.MU = parameterMu
        self.M = parameterM
        self.RHO = density
        self.G = gravity
        self.ALPHA = angleInDegrees / 180 * np.pi
        
        self.T = self.H / self.Q
        
        self.x_s = x_s
        self.x_f = x_f
        
        self.THETA = self.RHO * self.G * self.H * np.sin(self.ALPHA)
        self.KAPPA = 2 * self.H**2 / (self.L * self.Q) * self.MU * self.THETA**(self.M)
        self.LAMBDA = 2 * self.H**2 / (self.L * self.Q * (self.M + 2)) * self.MU * self.THETA**(self.M)
        
        self.h = None
        self.flowCalculated = False

    def generateLinearQ(self):
        dq = 2*(self.x_f + self.LAMBDA*self.h_0**(self.M+2)) / ((self.x_f  - self.x_s)**2)
        self.dq = dq
        
        self.fun_q = linear_q(dq, self.x_s, self.x_f)
        self.int_q = linear_int_q(dq, self.x_s, self.x_f)
        
    def setQ(self, q):
        self.fun_q = q
        
        num = 10000
        dx = 1/10000
        dq_on_grid = [(q(x) + q(x+dx))/2*dx for x in np.linspace(0, 1, num = num)]
        self.int_q_approx = np.cumsum(dq_on_grid)
        self.int_q = lambda x: np.interp(x, np.linspace(0, 1, num = 10000), self.int_q_approx)
        
    def calculateHeight(self):
        
        x = np.linspace(0, 1, num = 10001)
        core= self.h_0**(self.M+2) + [self.int_q(xi) for xi in x] / (self.LAMBDA)
        core[core < 0] = 0 

        h = np.power(core, 1/(self.M+2))
        self.h = h
        self.h_approx = lambda x: np.interp(x, np.linspace(0, 1, num=10001), self.h)
        
    def calculateFlow(self):
        h_max = np.max(self.h)
        xx, zz = np.meshgrid(np.linspace(0., 1., num=50), np.linspace(0.,h_max, num=20))
        h_approx = lambda x: np.interp(x, np.linspace(0, 1, num=10001), self.h)
        
        valid1 = h_approx(xx) > 0
        valid2 = zz < h_approx(xx)
        valid = np.logical_and(valid1, valid2)
        
#        xx = xx[valid]
#        zz = zz[valid]
        
        q = np.vectorize(self.fun_q)
        
        u = np.zeros(xx.shape)
        v = np.zeros(xx.shape)
        
        xx_valid = xx[valid]
        zz_valid = zz[valid]
        
        u[valid] = self.KAPPA * (h_approx(xx_valid)**(self.M+1) - (h_approx(xx_valid) - zz_valid)**(self.M+1))/(self.M+1)
        v[valid] = (((1 - zz_valid / h_approx(xx_valid))**(self.M))-1)/h_approx(xx_valid)*(q(xx_valid))*(self.M+2)/self.KAPPA
        
        self.xx = xx
        self.zz = zz
        self.u = u
        self.v = v
        self.flowCalculated = True
        
    def calculatePath(self, x0, T_final=1.0):
        if  np.any(self.h == None):
            self.calculateHeight()
        
        if not(self.flowCalculated):
            self.calculateFlow()
        
        h_approx = lambda x: np.interp(x, np.linspace(0, 1, num = 10001), self.h)
        
        num = 10000
        x = np.zeros((num, 2))
        x[0] = x0
        dt = T_final / num
        for i in range(num-1):
            if (x[i, 0] > 1 or x[i, 1] > h_approx(x[i, 0]) or i == num) :
                x = x[:i]
                T_final = i * dt
                break
            u = self.KAPPA * (h_approx(x[i, 0])**(self.M+1) - (h_approx(x[i,0]) - x[i,1])**(self.M+1))/(self.M+1)
            v = -(1 - ((1 - x[i,1] / h_approx(x[i,0]))**(self.M)))/h_approx(x[i,0])*(self.fun_q(x[i,0]))
            x[i+1, 0] = x[i,0] + u * dt
            x[i+1, 1] = x[i,1] + v * dt
#        x = x[::500,]
        plt.text(x[-1, 0]*self.L, x[-1, 1]*self.H, "T={:.1f} yr".format(T_final*self.T/(24*365*3600)))
        plt.plot(x[:, 0]*self.L, x[:, 1]*self.H, color='tab:green')
        
    def getHeight(self, x):
        """Returns the height of the glacier at unscaled location x"""
        if  np.any(self.h == None):
            self.calculateHeight()
        return self.h_approx(x)
        
    def plotQ(self, x = np.linspace(0, 1, num = 10001), plotHandle = plt):
        plotHandle.plot(x*self.L, [self.fun_q(xi)*self.Q for xi in x]) # 24*3600*365*100
        plotHandle.plot(x*self.L, [self.int_q(xi)*self.Q for xi in x]) # 24*3600*365*100
        plotHandle.plot((0, self.L), (-self.Q * self.LAMBDA * self.h_0**(self.M+2), -self.Q*self.LAMBDA * self.h_0**(self.M+2)), alpha=0.3)
        plotHandle.set_ylabel('Accumulation (m/s)')
        plotHandle.set_xlabel('Length(m)')
        plotHandle.legend(['$q^*(x)$', '$\int q^*(x) dx$', '$-Q \lambda h_0^{M+2}$'])
        
    def plotGlacier(self, x = np.linspace(0, 1, num = 10001), plotHandle = plt,
                    plotFlow = True):
        
        if  np.any(self.h == None):
            self.calculateHeight()
        
        if not(self.flowCalculated):
            self.calculateFlow()
        
#        plotHandle.ylabel('$h^*$'); 
#        plotHandle.xlabel('$x^*$'); 
#        plotHandle.ylim(((np.min(self.h) - 0.1)*self.H, self.H*(np.max(self.h)+0.1)))
#        plotHandle.axis('equal')
        plotHandle.set_xlabel('Length (m)')
        plotHandle.set_ylabel('Height (m)')
        
#        print(np.max(self.u)*(self.L * self.Q / self.H))
        
        speed = (self.L * self.Q / self.H)**2*self.u**2 + (self.Q)**2*self.v**2
#        print(self.xx.shape)
#        colors = np.matrix([[0., 0., speedi] for speedi in speed])
#        print(colors.shape)
        
#        JOBBE VIDERE HER FOR Å FÅ ET FARGEPLOTT
#        SÅ KJØRE NYE TING MED RHO = 910
        
        
#        speed = np.reshape(np.shape(self.xx))
#        print(speed)
        
        plotHandle.plot(self.L*x, self.H*self.h)
        if plotFlow:
#            A = plt.contourf(self.xx*self.L, self.zz*self.H, speed, cmap="plasma", alpha=0.2)
#            plt.colorbar()
            plotHandle.quiver((self.xx)*self.L, (self.zz)*self.H, self.u*(self.L * self.Q / self.H), self.v*self.Q, speed,
                              angles = 'xy', pivot = 'middle', scale = 1E-4, cmap = 'inferno_r')
#            plotHandle.colorbar(A)


def linear_q(dq, x_s, x_f):
    def q(x):
        result = 0.
        if x <= 0.:
            result = 0.
        elif x < x_s:
            result = 1.
        elif x < x_f:
            result = (1. - dq*(x - x_s))
        elif x >= x_f:
            result = 0.
        return result
    return q


def linear_int_q(dq, x_s, x_f):
    def int_q(x):
        if x <= 0:
            return 0
        elif x < x_s:
            return x
        elif x < x_f:
            return x  - dq / 2 * (x - x_s)**2
        else:
            return x_f - dq / 2 * (x_f - x_s)**2
    return int_q


if __name__  == "__main__":
    q_bergen = lambda x: .25-1.0*x**2
    #q_bergen = lambda x: np.arctan(-x)
    
        
    q_0 = linear_q(6., 0.33, 1.0)
    
    
    #fig, (ax1, ax2) = plt.subplots(2, 1)
    
    #G_bergen = StationaryGlacier(50, 0.0, 2000, 2.0, 9.3E-25, 3, 1000, 9.81, 25.0, 0.60 ,0.9)
    #G_bergen.setQ(q_0)
    #G_bergen.plotQ(plotHandle = ax1)
    #G_bergen.plotGlacier(plotHandle = ax2)
    #fig.savefig("stationary_glacier.pdf")
    #plt.show()
    
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    G_linear = StationaryGlacier(100, .5, 6000, 2.0, 9.3E-25, 3, 1000, 9.81, 20.0, 1/3 ,2/3)
    G_linear.generateLinearQ()
    G_linear.plotQ(plotHandle = ax1)
    G_linear.calculateHeight()
    G_linear.plotGlacier(plotHandle = ax2)
    x_start = 0.15
    G_linear.calculatePath(x0=(x_start, G_linear.h_approx(x_start)), T_final=3.0)
    fig.savefig("stationary_linear_glacier.pdf")


