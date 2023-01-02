import sympy as sp
import numpy as np
import math
from gekko import GEKKO
import matplotlib.pyplot as plt
import time
import random

# initialize GEKKO
m = GEKKO()
nt = 101
t_low = 0
t_high = 4
t_ = np.linspace(t_low, t_high,nt)
dt = (t_high-t_low)/(nt-1)
m.time = t_

# define weight matrix Q 
a = 1
b = 1
Qp = [[a, 0],[0, b]]

# define states
obs = [m.Param(value=0), m.Param(value=0)]  # observed state variables x and x_dot
x = [m.Var(value=0), m.Var(value=0)] # state variables x and x_dot
# this might be a decision variable

# gravity constant
gc = 9.81

# define parameter matrix as fixed variables of model
p= [m.FV(value=25.1), m.FV(value=1.2)] 

# define control 
# control input decision variable
u = m.MV(value=0, lb=0, ub=40)

# define dynamic model
v = [m.Intermediate(x[1]), m.Intermediate(-u*m.cos(x[0])/p[0]-gc*m.sin(x[0])-p[1]*x[1])] 

# define lp
lp = m.Param(value=0)
m.Equation(lp == (x[0]-obs[0])**2 + (x[1]-obs[1])**2)

# define equation (5), g_dot and g
g = [[m.Var(value=0), m.Var(value=0)], [m.Var(value=0), m.Var(value=0)]]
df_dx       = m.Intermediate([[0, u*m.sin(x[0])/p[0] -gc*m.cos(x[0])],[1, -p[1]]])
df_dp       = m.Intermediate([[0,2*u*m.cos(x[0])/p[0]**2], [0,-x[1]]])
g_dot       = m.Equation(np.dot(g, df_dx) + df_dp)
g           = m.Equation(m.integral(g_dot))

# define (7), h_dot and h
h = [[[m.Var(value=0), m.Var(value=0)], [m.Var(value=0), m.Var(value=0)]],[[m.Var(value=0), m.Var(value=0)],[m.Var(value=0), m.Var(value=0)]]]
d2f_dx2     = m.Intermediate([[[0, u*m.cos(x[0])/p[0]+gc*m.sin(x[0])],[0,0]],[[0,0],[0,0]]])
d2f_dpdx    = m.Intermediate([[[0, -2*u*m.sin(x[0])/p[0]**2],[0,0]],[[0,0],[0,-1]]])
df_dx       = m.Intermediate([[0, u*m.sin(x[0])/p[0] -gc*m.cos(x[0])],[1, -p[1]]])
d2f_dp2     = m.Intermediate([[[0, u*m.cos(x[0])/p[0]+gc*m.sin(x[0])],[0,0]],[[0,0],[0,0]]])
h_dot       = m.Equation(np.transpose(np.dot(np.transpose(np.dot(d2f_dx2, g)+d2f_dpdx,(0,2,1)), g),(0,2,1)) + np.dot(df_dx, h) + np.dot(d2f_dpdx, g) + d2f_dp2)
h           = m.Equation(m.integral(h_dot))

dx_lp = m.Intermediate([[2*(x[0]-obs[0])],[2*(x[1]-obs[1])]])
d2x_lp2 = m.Intermediate([[2],[2]])
d2jp_dp2 = m.Equation(m.integral(np.dot(np.transpose(np.dot(d2x_lp2, g)),  g) + np.dot(dx_lp, h)))

# get eigenvalues
eig = m.Equation(np.linalg.eigvals(d2jp_dp2)) #problem

# make list of all eigenvalues
eig_list = []
for i in range(0, len(eig)):
    eig_list.append(eig[i])

eig_list.sort()
max_eig = eig_list[-1]
min_eig = eig_list[0]

m.Intermediate(max_eig)
m.Intermediate(min_eig)
# define cost function
m.Obj(0.5*(max_eig/min_eig)**2)