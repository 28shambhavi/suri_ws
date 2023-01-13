import dynamics as dyn
import numpy as np
import pdb

def rollout():
    x_trj, obsx_trj = dyn.main()
    return x_trj, obsx_trj

def h_from_obj(x,x_obs, u, g, h, p):
    # compute the cost function
    gc = 9.81
    dt = 0.1 # what should this be???
    #x_obs = dyn.model_dyn(0,u,x,30,1)
    # print("\n h passed", h, "\n")
    # df_dx = np.transpose([[0, u*np.sin(x[0])/p[0]] -gc*np.cos(x[0]), [1, -p[1]]])
    df_dx = [[0,1],[u*np.sin(x[0])/p[0]-gc*np.cos(x[0]),  -p[1]]]
    # df_dp = np.transpose([[0,2*u*np.cos(x[0])/p[0]], [0, -x[1]]])
    df_dp = [[0,0],[2*u*np.cos(x[0])/(p[0])**2,-x[1]]]
    g_dot = np.dot(df_dx, g) + df_dp
    g = g + dt*g_dot
    # print("\n\n\n\n\n printing g", g)
    
    d2f_dx2 = [[[0, 0], [gc*np.sin(x[0])+u*np.cos(x[0])/p[0], 0]], [[0, 0], [0, 0]]]
    
    d2f_dpdx = [[[0, 0], [-2*u*np.sin(x[0])/p[0]**2, 0]], [[0, 0], [0, -1]]]
    
    d2f_dp2 = [[[0, 0], [-6*u*np.cos(x[0])/p[0]**3, 0]], [[0, 0], [0, 0]]]

    h_dot = np.transpose(np.dot(np.transpose(np.dot(d2f_dx2, g)+d2f_dpdx,(0,2,1)), g),(0,2,1)) + np.dot(df_dx, h) + np.dot(d2f_dpdx, g) + d2f_dp2
    # h_a = np.dot(np.transpose(np.dot(d2f_dx2,g)+d2f_dpdx,(0,2,1)),g)
    # h_b = np.dot(df_dx,h)+np.dot(d2f_dpdx,g)+d2f_dp2
    # h_dot =0
    print("\nh dot", h_dot, "\n")
    h = h + dt*h_dot

    # dx_lp = np.transpose(np.dot([[2,0],[0,2]],[[x[0]-x_obs[0]], [x[1]-x_obs[1]]]))
    dx_lp = [[2*(x[0]-x_obs[0]), 2*(x[1]-x_obs[1])]]
    d2x_lp2 = [[2,0],[0,2]]
    
    hessian_to_add = np.dot(np.transpose(np.dot(d2x_lp2,g)),g)+np.dot(dx_lp,h) #????????
    # print(hessian_to_add, "\n")
    return hessian_to_add
    # eig = np.linalg.eigvals(hessian)
    # print(eig)
    # eig_list = []
    # for i in range(0, len(eig)):
    #     eig_list.append(eig[i])

    # print(eig_list)
    # eig_list.sort()
    # max_eig = eig_list[-1]
    # min_eig = eig_list[0]

    # return 0.5*(max_eig/min_eig)**2

def cost_given_hsum(hsum):
    eig = np.linalg.eigvals(hsum)
    print(eig)
    [eig] = eig
    eig_list = []
    for i in range(0, len(eig)):
        if eig[i]!=0.00000:
            eig_list.append(eig[i])
    eig_list.sort()
    print("sorted eigenlist without zeroes", eig_list)
    max_eig = eig_list[-1]
    min_eig = eig_list[0]
    print("condition number", max_eig/min_eig)
    return 0.5*(max_eig/min_eig)**2

def main():
    x, x_obs = rollout()
    t_ = np.linspace(0,4,101)
    u1_sigma = 0.4
    u1_mu = 2
    u2_sigma = 0.4
    u2_mu = 0.8
    u3_sigma = 0.4
    u3_mu = 3
    u_init = 200*np.exp(-0.5*((t_-u1_mu)/u1_sigma)**2)/(u1_sigma*np.sqrt(2*np.pi))+150*np.exp(-0.5*((t_-u3_mu)/u3_sigma)**2)/(u3_sigma*np.sqrt(2*np.pi))+100*np.exp(-0.5*((t_-u2_mu)/u2_sigma)**2)/(u2_sigma*np.sqrt(2*np.pi)) -100
    u_orig = 40*np.exp(-0.5*((t_-u1_mu)/u1_sigma)**2)/(u1_sigma*np.sqrt(2*np.pi))
    u_trj = u_init
    p = [25.1, 1.2]
    g_trj = np.zeros((len(x),2,2))
    h_trj = np.zeros((len(x),2,2,2))
    h_over_traj = []
    h_in_cost = 0
    for i in range(1, len(x)):
        new_h = h_from_obj(x[i], x_obs[i], u_trj[i], g_trj[i-1,:,:], h_trj[i-1,:,:,:], p)
        h_in_cost+= new_h
        h_over_traj.append(new_h)
    print(h_in_cost)
    print(cost_given_hsum(h_in_cost))
        

if __name__ == '__main__':
    main()