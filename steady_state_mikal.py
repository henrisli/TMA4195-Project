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

    def generateLinearQ(self):
        dq = 2*(self.x_f + self.LAMBDA*self.h_0**(self.M+2)) / ((self.x_f  - self.x_s)**2)
        self.dq = dq
        
        self.fun_q = linear_q(dq, self.x_s, self.x_f)
        self.int_q = linear_int_q(dq, self.x_s, self.x_f)
        
    def setQ(self, q):
        self.fun_q = q
        
        dq_on_grid = [q(x)/1001 for x in np.linspace(0, 1, num = 1001)]
        self.int_q_approx = np.cumsum(dq_on_grid)
        self.int_q = lambda x: np.interp(x, np.linspace(0, 1, num = 1001), self.int_q_approx)
        
    def calculateHeight(self, x = np.linspace(0, 1, num = 1001)):
        core= self.h_0**(self.M+2) + [self.int_q(xi) for xi in x] / (self.LAMBDA) 
        core[core < 0] = 0 

        h = np.power(core, 1/(self.M+2))
        self.h = h
        
    def calculateFlow(self):
        xx, zz = np.meshgrid(np.linspace(0, 1, num=11), np.linspace(0,1, num=11))
        h_approx = lambda x: np.interp(x, np.linspace(0, 1, num=1001), self.h)
        
        valid1 = h_approx(xx) > 0
        valid2 = zz < h_approx(xx)
        valid = np.logical_and(valid1, valid2)

        
        q = np.vectorize(self.fun_q)
        qq = q(xx)
        
        xx = xx[valid]
        zz = zz[valid]
        
#        qq = [self.fun_q(xi) for xi in xx]
        print(qq)
        
#        DET SKJER NOE RART HER MED QQ
        
        u = self.KAPPA * (h_approx(xx)**(self.M+1) - (h_approx(xx) - zz)**(self.M+1))/(self.M+1)
        v = (1 - ((1 - zz / h_approx(xx))**(self.M)))/h_approx(xx)*(q(xx))
        
#        print(v)
        
        self.u = u
        self.v = v
        
        plt.quiver(xx*self.L, zz*self.H, u*(self.L * self.Q / self.H), v*self.Q)
        
        
    def plotQ(self, x = np.linspace(0, 1, num = 1001), plotHandle = plt):
        plotHandle.plot(x, [self.fun_q(xi) for xi in x])
        plotHandle.plot(x, [self.int_q(xi) for xi in x])
        plotHandle.plot((0, 1), (-self.LAMBDA * self.h_0**(self.M+2), -self.LAMBDA * self.h_0**(self.M+2)), alpha=0.3)
        
    def plotGlacier(self, x = np.linspace(0, 1, num = 1001), plotHandle = plt):
        self.calculateHeight(x)
        
        plotHandle.ylabel('$h^*$'); plt.xlabel('$x^*$'); plt.ylim((-self.H, 2*self.H))
#        plotHandle.axis('equal')
        plotHandle.plot(self.L*x, self.H*self.h)


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



#legend = []
#differentDownpour = [1.]
#for dp in differentDownpour:
#    G = StationaryGlacier(50, 1.0, 1000, dp, 9.3E-25, 3, 1000, 9.81, 5, 0.2 ,0.8)
#    G.generateLinearQ()
##    G.plotQ()
##    plt.show()
#    G.plotGlacier()
#    legend.append('Downpour = {}m/yr'.format(dp))
#plt.legend(legend)
    

q_bergen = lambda x: 0. if x < 0.2 else ( - x/50)
    
q_0 = linear_q(8.79, 0.2, 0.8)
iq = linear_int_q(8.79, 0.2, 0.8)
#print(iq(0.9))

#G_1 = StationaryGlacier(50, 1.0, 1000, 3.0, 9.3E-25, 3, 1000, 9.81, 5, 0.2 ,0.8)
##print(G_1.LAMBDA)
#G_1.setQ(q_0)
#G_1.plotQ()
#plt.show()
#G_1.plotGlacier()
#plt.show()
#
#G_2 = StationaryGlacier(50, 1.0, 1000, 3.0, 9.3E-25, 3, 1000, 9.81, 5, 0.2 ,0.8)
#G_2.generateLinearQ()
#G_2.plotQ()
#plt.show()
#G_2.plotGlacier()

G_bergen = StationaryGlacier(50, 1.0, 1000, 1., 9.3E-25, 3, 1000, 9.81, 5, 0.2 ,0.8)
G_bergen.setQ(q_bergen)
G_bergen.plotGlacier()
G_bergen.calculateFlow()

#G_1.plotGlacier()


