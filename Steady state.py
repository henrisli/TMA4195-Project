# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:11:01 2018

@author: henri
"""
import numpy as np
import matplotlib.pyplot as plt

H = 50
L = 1000
Q = 7.5/(365*24*3600)
mu = 9.3e-25
m = 3
rho = 1000
g = 9.81
alpha = 5*np.pi/180
Theta = rho*g*H*np.sin(alpha)

l = 3
dx = 1/100
kappa = 2*H**2/(Q*L)*mu*Theta**m
q = np.repeat(1,(l+l*l/2)/dx)
q = np.append(q,np.linspace(1,-1-l,(2+l)/dx))
q = np.append(q,np.repeat(0,4/dx))

h = np.zeros(len(q))
for i in range(len(q)):
    h[i] = np.power((m+2)/kappa*dx*np.sum(q[:i]),1/(m+2))

print(h)
plt.plot(h)