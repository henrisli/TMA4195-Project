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
        
        self.THETA = self.RHO * self.G * self.H * np.sin(self.ALPHA)
        self.LAMBDA = 2 * self.H**2 / (self.L * (self.M + 2)) * self.MU * self.THETA**(self.M)

        self.x_s = x_s
        self.x_f = x_f
        self.dq = 2*(self.Q*x_f + self.LAMBDA*self.h_0**(self.M+2)) / ((self.x_f  - self.x_s)**2)
        
        self.q = generate_q(self.Q, self.dq, self.x_s, self.x_f)
        self.int_q = generate_int_q(self.Q, self.dq, self.x_s, self.x_f)
        
    def plotQ(self, x = np.linspace(0, 1, num = 1001)):
        plt.plot(x, self.q(x))
        plt.plot(x, self.int_q(x))
        plt.plot((0, 1), (-self.LAMBDA * self.h_0**(self.M+2), -self.LAMBDA * self.h_0**(self.M+2)), alpha=0.3)
        
    def plotGlacier(self, x = np.linspace(0, 1, num = 1001)):
        core= 1 / (self.LAMBDA) * self.int_q(x) + self.h_0**(self.M + 2)
        
        assert(not(np.any(core < -1E-11)))
        core[core < 0] = 0 

        h = np.power(core, 1/(self.M+2))
        plt.ylabel('$h^*$'); plt.xlabel('$x^*$'); plt.ylim((0, 5*self.H))
        plt.axis('equal')
        plt.plot(self.L*x, self.H*h)
        
        

def generate_q(Q, dq, x_s, x_f):
    def f(x):
        if x < 0:
            return 0
        elif x < x_s:
            return Q
        elif x < x_f:
            return Q - dq*(x - x_s)
        else:
            return 0
    return np.vectorize(f)

def generate_int_q(Q, dq, x_s, x_f):
    def int_q(x):
        if x < 0:
            return 0
        elif x < x_s:
            return x*Q
        elif x < x_f:
            return Q*x  - dq / 2 * (x - x_s)**2
        else:
            return Q*x_f - dq / 2 * (x_f - x_s)**2
    int_q = np.vectorize(int_q)
    return int_q

legend = []
differentDownpour = [2.0, 10. ,70.]
for dp in differentDownpour:
    G = StationaryGlacier(50, 4.0, 1000, dp, 9.3E-25, 3, 1000, 9.81, 5, 0.2 ,0.8)
#    G.plotQ()
    G.plotGlacier()
    legend.append('Downpour = {}m/yr'.format(dp))
plt.legend(legend)
    
#G_1 = StationaryGlacier(50, 4.0, 1000, 3.0, 9.3E-25, 3, 1000, 9.81, 5, 0.2 ,0.4)
#G_1.plotGlacier()


