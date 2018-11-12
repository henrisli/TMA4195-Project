# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:11:01 2018

@author: henri
"""
import numpy as np
import matplotlib.pyplot as plt



H = 50
L = 2000
Q = 11/(365*24*3600)
mu = 9.3e-25
m = 3
rho = 1000
g = 9.81
alpha = 25*np.pi/180
Theta = rho*g*H*np.sin(alpha)

def production(h,*args):
    n = len(h) - 2
    q = np.zeros(n + 2)
    for i in range(n + 2):
        if i < n/3 + 1:
            q[i] = 1
        else:
            q[i] = 1-(i-(n/3+1))/(n/6)   
            if np.cumsum(q)[-1]<0:
                q[i] = 0
            
        if h[i]==0 and q[i]<0:
            q[i] = 0
    return q

dx = 1/150
kappa = 2*H**2/(Q*L)*mu*Theta**m
x = np.arange(-0.5*dx,1+1.5*dx,dx)
print(x)
h0 = np.ones(len(x))
q = production(h0)
qsum = np.abs(np.cumsum(q))
print((q[56]-q[57])/dx)

plt.plot(x,q)
plt.figure()
plt.plot(x,qsum*dx)
h = np.power((m+2)/kappa*dx*np.cumsum(q),1/(m+2))
plt.figure()
plt.plot(x,h*H)
plt.plot(x,np.zeros(len(x)))