# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:11:01 2018

@author: mikalst
"""
import numpy as np
import matplotlib.pyplot as plt

#CONSTANTS
H = 50
h_0 = 1.0 #Initial height is 100% of height scale
L = 1000
Q = 7.5/(24*365*3600) #7.5m per year
MU = 9.3e-25
M = 3
RHO = 1000
G = 9.81
ALPHA = 5*np.pi/180
THETA = RHO*G*H*np.sin(ALPHA)
KAPPA = 2*H**2/(Q*L)*MU*THETA**M

print('Kappa: {}'.format(KAPPA))
print('Theta: {}'.format(THETA))

LAMBDA = Q*KAPPA / (M+2)

X_BREAK = 0.2


def integrated_q(x, x_b, descent = None):
    if descent == None:
        descent = Q / (x_b)
    if x < 0:
        return 0
    elif x < x_b:
        return x*Q
    else:
        return x*Q  - descent / 2 * (x**2 + x_b**2) + descent*x*x_b
integrated_q = np.vectorize(integrated_q)
int_q = lambda x: integrated_q(x, X_BREAK, Q/X_BREAK)

def core(x):
    return max(0, 1/LAMBDA*int_q(x) + h_0**(M+2))
core = np.vectorize(core)

x = np.linspace(0, 1, num=1001)

plt.plot(x, int_q(x))
plt.show()

h = np.power(core(x), 1/(M+2))

plt.plot(x, core(x))
plt.show()

plt.ylabel('$h^*$'); plt.xlabel('$x^*$'); plt.ylim((0, 5*H))
plt.axis('equal')
plt.plot(L*x, H*h)
