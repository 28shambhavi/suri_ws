import numpy as np
import cyipopt

# solve inverted pendulum problem using cyipopt

class Test_model:

    def __init__(self):
        pass

    def objective(self, variables):
        x = [variables[0], variables[1]] #state pos and vel
        params = [variables[2], variables[3]] #alpha and beta estimated
        control = variables[4] #control acceleration
        g = [[variables[5], variables[6]], [variables[7], variables[8]]] #g matrix
        h = [[[variables[9], variables[10]], [variables[11], variables[12]]], [[variables[13], variables[14]], [variables[15], variables[16]]]] #h matrix
        hessian = [[variables[17], variables[18]], [variables[19], variables[20]]] #hessian matrix
        
        gc = 9.81
        real_params = [30.0, 1.0] #real alpha and beta
        
        #define dynamics model
        v = [x[1], -gc*np.sin(x[0]) - params[1]*x[1] - control*np.cos(x[0])/params[0]]

        #calculate observed states
        dt = 0.1 # what should this be???
        x_obs = [x[0] + dt*v[0], x[1] + dt*v[1]]

        lp = (x[0]-x_obs[0])**2 + (x[1]-x_obs[1])**2
        df_dx = [[0, control*np.sin(x[0])/params[0]] -gc*np.cos(x[0]), [1, -params[1]]]
        df_dp = [[0,2*control*np.cos(x[0])/params[0]], [0, -x[1]]]
        
        g_dot = np.dot(df_dx, g) + df_dp
        g = g + dt*g_dot

        d2f_dx2 = [[[0, control*np.cos(x[0])/params[0]]+gc*np.sin(x[0]), [0, 0]], [[0, 0], [0, 0]]]
        d2f_dpdx = [[[0, -2*control*np.sin(x[0])/params[0]**2], [0, 0]], [[0, 0], [0, -1]]]
        d2f_dp2 = [[[0, control*np.sin(x[0])/params[0]+gc*np.sin(x[0])], [0, 0]], [[0, 0], [0, 0]]]
        
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

    def gradient(self, x):
        pass
        
    def solve(self, x):
        #find descent direction
        hessian = 
        dx = -np.dot(np.linalg.inv(hessian), dx_lp)
        

    def intermediate(self,
                     alg_mod,
                     iter_count,
                     obj_value,
                     inf_pr,
                     inf_du,
                     mu,
                     d_norm,
                     regularization_size,
                     alpha_du,
                     alpha_pr,
                     ls_trials):
        pass

def main():
    # define variables
    # len of x0 = 16
    x0 = np.array([0,0,25.1,1.2, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]) # x, v, alpha, beta, control

    # lower and upper bounds
    # lb = np.array([0, 0, 0, 0])
    # ub = np.array([1, 1, 1, 1])

    # create instance of the class
    nlp = cyipopt.Problem(
        n=len(x0),
        m=0,
        problem_obj=Test_model(),
        # lb=lb,
        # ub=ub,
    )

    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)
    nlp.add_option('hessian_approximation', 'limited-memory')

    x, info = nlp.solve(x0)

    print(x)
    print(info)

if __name__ == '__main__':
    main()

