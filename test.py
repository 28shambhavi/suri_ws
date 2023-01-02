import numpy as np
import math
from gekko import GEKKO
import matplotlib.pyplot as plt
import time
import random
import sympy as sp

# initialize GEKKO
m = GEKKO()
nt = 101
t_low = 0
t_high = 4
t_ = np.linspace(t_low, t_high,nt)
dt = (t_high-t_low)/(nt-1)
m.time = t_

# Variables
x = [m.Var(value=0), m.Var(value=0)] # state variables x and x_dot

# optimize final time
tf = m.FV(value=4, lb=0, ub=4)
tf.STATUS = 1

# control changes every time period
u = m.MV(value=0, lb=0, ub=40)
u.STATUS = 1

# define the ODEs that describe the movement of the vehicle
m.Equation(x[0].dt()==x[1]*tf)
m.Equation(x[1].dt()==u*tf)

# define path constraints

# define voundary constraints

# Objective function
m.Obj(tf)

# Solve
m.options.IMODE = 6
m.solve(disp=True) # set to True to view solver logs

#Presentation of results
print('Final Time: ' + str(tf.value[0]))

# plot results
plt.figure()
plt.plot(t_, x[0].value, 'k-', label=r'$x$')
plt.plot(t_, x[1].value, 'b-', label=r'$x_dot$')
plt.plot(t_, u.value, 'g-', label=r'$u$')
plt.legend(loc='best')
plt.show()