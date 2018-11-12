# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:11:01 2018

@author: mikalst
"""
import numpy as np
import matplotlib.pyplot as plt

class StationaryGlacier:
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
        
    def calculateHeight(self, x = np.linspace(0, 1, num = 1001)):
        core= self.h_0**(self.M+2) + [self.int_q(xi) for xi in x] / (self.LAMBDA)
        core[core < 0] = 0 

        h = np.power(core, 1/(self.M+2))
        self.h = h
        
    def calculateFlow(self):
        h_max = np.max(self.h)
        xx, zz = np.meshgrid(np.linspace(0, 1, num=11), np.linspace(0,h_max, num=11))
        h_approx = lambda x: np.interp(x, np.linspace(0, 1, num=1001), self.h)
        
        valid1 = h_approx(xx) > 0
        valid2 = zz < h_approx(xx)
        valid = np.logical_and(valid1, valid2)
        
        xx = xx[valid]
        zz = zz[valid]
        
        q = np.vectorize(self.fun_q)
        
        u = self.KAPPA * (h_approx(xx)**(self.M+1) - (h_approx(xx) - zz)**(self.M+1))/(self.M+1)
        v = (1 - ((1 - zz / h_approx(xx))**(self.M)))/h_approx(xx)*(q(xx))
        
        self.xx = xx
        self.zz = zz
        self.u = u
        self.v = v
        
        
    def plotQ(self, x = np.linspace(0, 1, num = 1001), plotHandle = plt):
        plotHandle.plot(x*self.L, [self.fun_q(xi)*self.Q for xi in x]) # 24*3600*365*100
        plotHandle.plot(x*self.L, [self.int_q(xi)*self.Q for xi in x]) # 24*3600*365*100
        plotHandle.plot((0, self.L), (-self.Q * self.LAMBDA * self.h_0**(self.M+2), -self.Q*self.LAMBDA * self.h_0**(self.M+2)), alpha=0.3)
        plotHandle.legend(['$q^*(x)$', '$\int q^*(x) dx$', '$-Q \lambda h_0^{M+2}$'])
        
    def plotGlacier(self, x = np.linspace(0, 1, num = 1001), plotHandle = plt,
                    plotFlow = True):
        
        if self.h == None:
            self.calculateHeight(x)
        
        if not(self.flowCalculated):
            self.calculateFlow()
        
#        plotHandle.ylabel('$h^*$'); 
#        plotHandle.xlabel('$x^*$'); 
#        plotHandle.ylim(((np.min(self.h) - 0.1)*self.H, self.H*(np.max(self.h)+0.1)))
#        plotHandle.axis('equal')
        print(np.max(self.u)*(self.L * self.Q / self.H))
        plotHandle.plot(self.L*x, self.H*self.h)
        if plotFlow:
            plotHandle.quiver(self.xx*self.L, self.zz*self.H, self.u*(self.L * self.Q / self.H), self.v*self.Q)


def linear_q(dq, x_s, x_f):
    def q(x):
        result = 0
        if x <= 0:
            result = 0
        elif x < x_s:
            result = 1
        elif x < x_f:
            result = (1 - dq*(x - x_s))
        elif x >= x_f:
            result = 0
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


q_bergen = lambda x: .25-1.0*x**2
#q_bergen = lambda x: np.arctan(-x)

    
q_0 = linear_q(20., 0.60, 0.90)


fig, (ax1, ax2) = plt.subplots(2, 1)

G_bergen = StationaryGlacier(50, 0.0, 2000, 2.0, 9.3E-25, 3, 1000, 9.81, 25.0, 0.60 ,0.9)
G_bergen.setQ(q_0)
G_bergen.plotQ(plotHandle = ax1)
G_bergen.plotGlacier(plotHandle = ax2)
fig.savefig("stationary_glacier.pdf")
plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1)

G_linear = StationaryGlacier(50, 0.0, 2000, 2.0, 9.3E-25, 3, 1000, 9.81, 25.0, 0.60 ,.9)
G_linear.generateLinearQ()
G_linear.plotQ(plotHandle = ax1)
G_linear.plotGlacier(plotHandle = ax2)
fig.savefig("stationary_linear_glacier.pdf")
#G_1.plotGlacier()


