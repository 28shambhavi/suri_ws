import numpy as np
import math
import matplotlib.pyplot as plt
import yaml

class Data_idx:
    t_ = np.linspace(0,4,101)
    gc = 9.8
    dt = 4/100

    def __init__(self, u, t_idx, time):
        self.t_idx = t_idx
        self.time = time
        self.u_init = u
        self.g = []
        self.h = []
        self.x = []
        self.obs_x = []

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
    u_sigma = 0.4
    u_mu = 2
    u_init = 40*np.exp(-0.5*((t_-u_mu)/u_sigma)**2)/(u_sigma*np.sqrt(2*np.pi))
    obs_x = [0.5,0.2]
    x = [0.1,0.1]
    state = []
    obs_state = []
    a = 30
    b = 1
    init_a = 25.1
    init_b = 1.2
    data_t_idx = []
    for t_idx in range(len(t_)):
        data_t_idx.append(Data_idx(u_init[t_idx], t_idx, t_[t_idx]))
    
    for d in data_t_idx:
        d.v = [d.x[1],-d.u_init*math.cos(d.x[0])/a-gc*math.sin(d.x[0])-b*d.x[1]]
        x1 = np.dot(d.v,dt) + d.x

    plt.show()
main()