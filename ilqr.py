# iterative linear quadratic regulator for nonlinear trajectory optimization

import numpy as np
import matplotlib.pyplot as plt
import scipy

class Test_model:

    def __init__(self):
        pass
    # define the system
    def f(self, x, p, u):
        gc = 9.81
        v = [x[1], -gc*np.sin(x[0]) - p[1]*x[1] - u*np.cos(x[0])/p[0]]
        return v

    # define the cost function
    def j_tau(self, x, p, u, g, h, hessian, dt):
        gc = 9.81
        v = self.f(x, p, u)
        x_obs = [x[0] + dt*v[0], x[1] + dt*v[1]]

        lp = (x[0]-x_obs[0])**2 + (x[1]-x_obs[1])**2
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
        hessian += np.dot(np.transpose(np.dot(d2x_lp2,g)),g)+np.dot(dx_lp,h)
        
        eig = np.linalg.eig(hessian)
        eig_list = []
        for i in range(0, len(eig)):
            eig_list.append(eig[i])

        eig_list.sort()
        max_eig = eig_list[-1]
        min_eig = eig_list[0]

        return 0.5*(max_eig/min_eig)**2
        

    # define the cost function gradient
    def j_tau_grad(self, x, p, u, g, h):
        pass

    # define the cost function hessian
    def j_tau_hess(self, x, p, u, g, h):
        pass

    def ilqr(self, f, j_tau, j_tau_grad, j_tau_hess, x0, p, u0, g, h, N, dt, max_iter):
        # initialize the trajectory
        x = np.zeros((N, x0.shape[0]))
        u = np.zeros((N, u0.shape[0]))
        x[0] = x0
        u[0] = u0
        # iterate
        for i in range(max_iter):
            # backward pass
            V = np.zeros((N, 1))
            Vx = np.zeros((N, x0.shape[0]))
            Vxx = np.zeros((N, x0.shape[0], x0.shape[0]))
            Vx[-1] = g[-1]
            Vxx[-1] = h[-1]
            for k in range(N-2, -1, -1):
                Qx = j_tau_grad(x[k], p, u[k], g[k], h[k])
                Qu = j_tau_grad(x[k], p, u[k], g[k], h[k])
                Qxx = j_tau_hess(x[k], p, u[k], g[k], h[k])
                Quu = j_tau_hess(x[k], p, u[k], g[k], h[k])
                Qux = j_tau_hess(x[k], p, u[k], g[k], h[k])
                # compute the feedback gain
                K = -np.dot(np.linalg.inv(Quu), Qux)
                # compute the value function gradient
                Vx[k] = Qx + np.dot(Qux.T, Vx[k+1])
                # compute the value function hessian
                Vxx[k] = Qxx + np.dot(np.dot(Qux.T, Vxx[k+1]), Qux)
                # compute the value function
                V[k] = j_tau(x[k], p, u[k], g[k], h[k]) + np.dot(Vx[k+1].T, np.dot(Vxx[k+1], Vx[k+1]))

            # forward pass
            for k in range(N-1):
                Qx = j_tau_grad(x[k], p, u[k], g[k], h[k])
                Qu = j_tau_grad(x[k], p, u[k], g[k], h[k])
                Qxx = j_tau_hess(x[k], p, u[k], g[k], h[k])
                Quu = j_tau_hess(x[k], p, u[k], g[k], h[k])
                Qux = j_tau_hess(x[k], p, u[k], g[k], h[k])
                # compute the feedback gain
                K = -np.dot(np.linalg.inv(Quu), Qux)
                # compute the control update
                du = -np.dot(K, Vx[k]) - np.dot(np.linalg.inv(Quu), Qu)
                # update the control
                u[k] = u[k] + du
                # update the state
                x[k+1] = f(x[k], p, u[k])
                # update the value function gradient
                Vx[k+1] = Qx + np.dot(Qux.T, Vx[k+1])
                # update the value function hessian
                Vxx[k+1] = Qxx + np.dot(np.dot(Qux.T, Vxx[k+1]), Qux)
                # update the value function
                V[k+1] = j_tau(x[k], p, u[k], g[k], h[k]) + np.dot(Vx[k+1].T, np.dot(Vxx[k+1], Vx[k+1]))
                


        return x, p, u, g, h

def main():
