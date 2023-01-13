# iterative linear quadratic regulator for nonlinear trajectory optimization

import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random

def continuous_dynamics(x, u, p):
    # define dynamic model
    gc = 9.81
    v = [x[1], -u*math.cos(x[0])/p[0]-gc*math.sin(x[0])-p[1]*x[1]]
    return v
    
def discrete_dynamics(x, u, p, dt):
    # define dynamic model
    gc = 9.81
    x = x + dt*continuous_dynamics(x,u,p)
    x_next = x
    return x_next

def rollout(x0, u_trj, p0):
    # rollout the trajectory
    dt = 0.1 #scalar time step
    x_trj = np.zeros((u_trj.shape[0]+1, x0.shape[0]))
    x_trj[0,:] = x0
    for i in range(0,x_trj.shape[0]-1): 
      x_trj[i+1,:] = discrete_dynamics(x_trj[i,:],u_trj[i,:], p0, dt)
    return x_trj

def cost_function(x, u, g, h, p, Q, R):
    # compute the cost function
    gc = 9.81
    dt = 0.1 # what should this be???
    x_obs = discrete_dynamics(x, u, p, dt)

    df_dx = [[0, u*np.sin(x[0])/p[0]] -gc*np.cos(x[0]), [1, -p[1]]]
    df_dp = [[0,2*u*np.cos(x[0])/p[0]], [0, -x[1]]]
    
    g_dot = np.dot(df_dx, g) + df_dp
    g = g + dt*g_dot

    d2f_dx2 = [[[0, u*np.cos(x[0])/p[0]]+gc*np.sin(x[0]), [0, 0]], [[0, 0], [0, 0]]]
    d2f_dpdx = [[[0, -2*u*np.sin(x[0])/p[0]**2], [0, 0]], [[0, 0], [0, -1]]]
    d2f_dp2 = [[[0, u*np.sin(x[0])/p[0]+gc*np.sin(x[0])], [0, 0]], [[0, 0], [0, 0]]]
    
    h_dot = np.transpose(np.dot(np.transpose(np.dot(d2f_dx2,g)+d2f_dpdx,(0,2,1)),g),(0,2,1)+np.dot(df_dx,h)+np.dot(d2f_dpdx,g)+d2f_dp2)
    h = h + dt*h_dot

    dx_lp = [2*(x[0]-x_obs[0]), 2*(x[1]-x_obs[1])]
    d2x_lp2 = [[2],[2]]
    hessian =0
    hessian += np.dot(np.transpose(np.dot(d2x_lp2,g)),g)+np.dot(dx_lp,h) #????????
    
    eig = np.linalg.eig(hessian)
    eig_list = []
    for i in range(0, len(eig)):
        eig_list.append(eig[i])

    eig_list.sort()
    max_eig = eig_list[-1]
    min_eig = eig_list[0]

    return 0.5*(max_eig/min_eig)**2

def cost_trj(x_trj, u_trj, g, h, p0, Q, R):
    cost = 0
    for i in range(0, x_trj.shape[0]-1):
        cost += cost_function(x_trj[i,:], u_trj[i,:], g[i,:], h[i,:], p0, Q, R)
    return cost

def Q_terms():
    pass

def gains(Q_uu, Q_u, Q_ux):
    k = np.zeros(Q_u.shape)
    K = np.zeros(Q_ux.shape)

    k = k-

def V_terms():
    pass

def expected_cost_reduction():
    pass

def forward_pass(x_trj, u_trj, p, k_trj, K_trj):
    x_trj_new = np.zeros(x_trj.shape)
    x_trj_new[0,:] = x_trj[0,:]
    u_trj_new = np.zeros(u_trj.shape)
    # TODO: Implement the forward pass here
#     for n in range(u_trj.shape[0]):
#         u_trj_new[n,:] = # Apply feedback law
#         x_trj_new[n+1,:] = # Apply dynamics
    dt =0.1
    for n in range(u_trj.shape[0]):
      u_trj_new[n,:] =u_trj[n,:]+k_trj[n,:] + K_trj[n,:].dot((x_trj_new[n,:]-x_trj[n,:])) # Apply feedback law
      x_trj_new[n+1,:] =discrete_dynamics(x_trj_new[n,:],u_trj_new[n,:], p[n, :], dt)
    return x_trj_new, u_trj_new


def backward_pass(x_trj, u_trj, regu):
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
    expected_cost_redu = 0
    
    
    for n in range(u_trj.shape[0]-1, -1, -1):
        
        Q_uu = 0
        k =0
        expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
    return k_trj, K_trj, expected_cost_redu

