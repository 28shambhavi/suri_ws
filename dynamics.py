import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random

def g_dot(t_idx, u, x, a, b, g):
    global gc, t_
    df_dx = [[0, u*math.sin(x[0])/a -gc*math.cos(x[0])],[1, -b]]
    df_dp = [[0,2*u*math.cos(x[0])/a**2], [0,-x[1]]]
    [g]=g
    g_dot_ = np.dot(df_dx,g) + df_dp
    # g[t_idx+1] = np.dot(g_dot_, dt) + g[t_idx]
    return g_dot_

def h_dot(t_idx, u, x, a, b, g, h):
    global gc, t_
    d2f_dx2 = [[[0, u*math.cos(x[0])/a+gc*math.sin(x[0])],[0,0]],[[0,0],[0,0]]]
    d2f_dpdx = [[[0, -2*u*math.sin(x[0])/a**2],[0,0]],[[0,0],[0,-1]]]
    df_dx = [[0, u*math.sin(x[0])/a -gc*math.cos(x[0])],[1, -b]]
    d2f_dp2 = [[[0, u*math.cos(x[0])/a+gc*math.sin(x[0])],[0,0]],[[0,0],[0,0]]]
    h_dot_ = np.transpose(np.dot(np.transpose(np.dot(d2f_dx2, g)+d2f_dpdx,(0,2,1)), g),(0,2,1)) + np.dot(df_dx, h) + np.dot(d2f_dpdx, g) + d2f_dp2
    return h_dot_

def hessian(obs_x, x, t_idx, a,b):
    global qp, c1, c2, u_init, t_, dt
    #print("obs x ",obs_x, "state x", x, "\n")
    dx_lp = [[2*c1*(x[0]-obs_x[0])],[2*c2*(x[1]-obs_x[1])]]
    d2x_lp2 = [[2*c1],[2*c2]]
    d2jp_dp2 = [[0,0],[0,0]]
    g = [([[0,0],[0,0]])]
    h = [([[0,0],[0,0]])]
    
    for i in range(t_idx):
        g.append(np.array(g)[2*i:2*i+2,:]+g_dot(i, u_init[i], x, a, b, np.array(g)[2*i:2*i+2,:])*dt)
        #print("\n", g, "\n")
        h.append(h[i] + h_dot(i, u_init[i], x, a, b, g[i], h[i])*dt)
        d2jp_dp2 += np.dot(np.transpose(np.dot(d2x_lp2, g[i])),  g[i])*dt + np.dot(dx_lp, h[i])  #hessian
    return d2jp_dp2

def model_dyn(t_idx, u, x, a, b):
    global gc, dt, t_
    print(x,"state x \n")
    v = [x[1],-u*math.cos(x[0])/a-gc*math.sin(x[0])-b*x[1]]
    x1 = np.dot(v,dt) + x
    return x1

def main():
    global dt, qp, gc,dt, g, h, c1, c2, u_init, t_
    c1 = 1
    c2 = 1
    t_ = np.linspace(0,4,101)
    dt = 4/100
    gc = 9.8 #constant parameter, g 
    g = [[0,0],[0,0]]
    qp = [[c1,0],[0,c2]]
    x = [[0],[0]]
    a = 30
    b = 25
    u1_sigma = 0.4
    u1_mu = 2
    u2_sigma = 0.4
    u2_mu = 0.8
    u3_sigma = 0.4
    u3_mu = 3
    u_init = 200*np.exp(-0.5*((t_-u1_mu)/u1_sigma)**2)/(u1_sigma*np.sqrt(2*np.pi))+150*np.exp(-0.5*((t_-u3_mu)/u3_sigma)**2)/(u3_sigma*np.sqrt(2*np.pi))+100*np.exp(-0.5*((t_-u2_mu)/u2_sigma)**2)/(u2_sigma*np.sqrt(2*np.pi)) -100
    u_orig = 40*np.exp(-0.5*((t_-u1_mu)/u1_sigma)**2)/(u1_sigma*np.sqrt(2*np.pi))
    obs_x = [0.5,0.2]
    x = [0.1,0.1]
    
 
    state = [[0,0]]
    obs_state = [[0,0]]
    a = 30
    b = 1
    init_a = 25.1
    init_b = 1.2
    for t_idx in range(len(t_)-1):
        state.append(np.array(model_dyn(t_[t_idx], u_orig[t_idx],np.array(state)[t_idx,:],init_a,init_b)))
        obs_state.append(np.array(model_dyn(t_[t_idx], u_orig[t_idx],np.array(obs_state)[t_idx,:]+np.random.normal(0, .004, 2),a,b)))
        
        
        #state_hessian=hessian(np.array(obs_state)[t_idx,:], np.array(state)[t_idx,:], t_idx, a,b)
    #g_dot(t_idx, u, t_, x, a, b)
    #x_state = np.array(state)[:,0]
    print(time.time())
    fig, (ax1, ax2) = plt.subplots(2)
    
    fig.tight_layout(pad=3)
    ax1.plot(t_, np.array(state)[:,0])
    ax1.plot(t_,np.array(obs_state)[:,0])
    ax1.set_xlabel('Time (in seconds)')
    ax1.set_ylabel('Position (in meters)')

    ax2.plot(t_, np.array(state)[:,1])
    ax2.plot(t_,np.array(obs_state)[:,1])
    ax2.set_xlabel('Time (in seconds)')
    ax2.set_ylabel('Velocity (in meters/second)')
    # plt.plot(t_, state)
    # plt.plot(t_, obs_state)
    # plt.plot(t_, u_init, "green")
    # plt.xlabel('Time (in seconds)')
    # plt.ylabel('Control Input (acc in m/s^2)')
    plt.show()
main()




# def lp(obs_x, x):
#     global qp
#     lp_ = np.dot(np.dot([x[0]-obs_x[0],x[1]-obs_x[1]], qp), np.transpose([x[0]-obs_x[0],x[1]-obs_x[1]]))
#     return lp_