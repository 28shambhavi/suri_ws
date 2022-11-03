import numpy as np
import math
import matplotlib.pyplot as plt
import time

t_ = np.linspace(0,4,101)
dt = 4/100
gc = 9.8 #gravity constant

def gaussian(x,a,m,s):
        return a*np.exp(-np.square(x-m)/(2*s*s))

def dynamics(t_idx, x, a, b, u):
    global gc, dt, t_
    print(x,"state x \n")
    v = [x[1],-u*math.cos(x[0])/a-gc*math.sin(x[0])-b*x[1]]
    x1 = np.dot(v,dt) + x
    return x1

def get_deriv(x,a,m,s,y,pred):
    pred_deriv = np.array([gaussian(x,a,m,s)/a,((x-m)/(s*s))*gaussian(x,a,m,s),(np.square(x-m)/(s*s*s))*gaussian(x,a,m,s)])
    return pred_deriv

def show_results(X,Y,pred,cf,save=None):
    l = np.arange(0,len(cf),1)
    fig = plt.figure(figsize = [25,10], dpi = 60)
    ax = fig.add_subplot(131)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.plot(X,Y,'r')
    ax.set_title('Input Distribution', fontsize = 30)
    ax1 = fig.add_subplot(132)
    ax1.tick_params(axis='x', labelsize=22)
    ax1.tick_params(axis='y', labelsize=22)
    ax1.plot(X,pred)
    ax1.set_title('Predicted Distribution', fontsize = 30)
    ax2 = fig.add_subplot(133)
    ax2.tick_params(axis='x', labelsize=22)
    ax2.tick_params(axis='y', labelsize=22)
    ax2.plot(l,cf)
    ax2.set_title('Objective Function vs No. of Iterations', fontsize = 25)
    if save:
        plt.savefig(str(save) + '.png')
    plt.show()

x = np.linspace(-25,25,100)
y = gaussian(x,10,0,20)

def gradient_descent(lr,k,itr,x,y,threshold):
    cf = []
    for i in range(itr):
        k_old = k
        r_ = 0.0
        for j in range(x.shape[0]):
            pred = gaussian(x[j],k[0],k[1],k[2])
            r = pred-y[j]
            r_ = r_ + r*r
            pred_deriv = get_deriv(x[j],k[0],k[1],k[2],y[j],pred)
            Jr = 2*(pred-y[j])*pred_deriv
            del_k = -lr*Jr
            k = k + del_k
        cf.append(r_)
        if(np.linalg.norm(k-k_old)<threshold):
#            print("quit")
            print("Converged at " + str(i) + "th iteration")
            break    
    return k, cf

lr = 0.01
k = np.ones(3)
threshold = 1e-15
itr = 1000
print("Baseline Results")
param_gd, cf_gd = gradient_descent(lr,k,itr,x,y,threshold)
print(param_gd)

print("\nExperiment 1 Plots: Different Initial Estimate ")
k_new = [30,35,40]
param_gd1, cf_gd1 = gradient_descent(lr,k_new,5000,x,y,threshold)
print(param_gd1)

print("\nExperiment 2 Plots: Different Number of Observations")
nx = np.linspace(-25,25,20)
ny = gaussian(nx,10,0,20)
param_gd2, cf_gd2 = gradient_descent(lr,k,5000,nx,ny,threshold)
print(param_gd2)

print("\nExperiment 3 Plots: Added Noise")
noise = np.random.normal(0.1,0.1,len(x))
noise_y = y + 1*noise
param_gd3, cf_gd3 = gradient_descent(lr,k,5000,x,noise_y,threshold)
print(param_gd3)

def gauss_newton(lr,k,itr,x,y,threshold):
    cf = []
    for i in range(itr):
        k_old = k
        r_ = 0.0
        for j in range(x.shape[0]):
            pred = gaussian(x[j],k[0],k[1],k[2])
            r = pred - y[j]
            r_ = r_ + r*r
            pred_deriv = get_deriv(x[j],k[0],k[1],k[2],y[j],pred)
            Jr = 2*(pred-y[j])*pred_deriv
            mag = np.linalg.norm(pred_deriv,ord=None)
            del_k = -lr*Jr/mag
            k = k + del_k
        cf.append(r_)
        if(np.linalg.norm(k-k_old)<threshold):
#            print("quit")
            print("Converged at " + str(i) + "th iteration")
            break    
    return k, cf
    
lr = 0.01
k = np.ones(3)
threshold = 1e-15
itr = 1000
print("\nBaseline Results")
param_gn, cf_gn = gauss_newton(lr,k,itr,x,y,threshold)
print(param_gn)

print("\nExperiment 1 Plots: Different Initial Estimate ")
param_gn1, cf_gn1 = gauss_newton(lr,k_new,5000,x,y,threshold)
print(param_gn1)

print("\nExperiment 2 Plots: Different Number of Observations")
param_gn2, cf_gn2 = gauss_newton(lr,k,5000,nx,ny,threshold)
print(param_gn2)

print("\nExperiment 3 Plots: Added Noise")
param_gn3, cf_gn3 = gauss_newton(lr,k,itr,x,noise_y,threshold)
print(param_gn3)

print("\nBaseline Plots\n")
print("a. Gradient Descent")
pred_gd = gaussian(x,param_gd[0],param_gd[1],param_gd[2])
show_results(x,y,pred_gd,cf_gd,9)
print("b. Gauss Newton")
pred_gn = gaussian(x,param_gn[0],param_gn[1],param_gn[2])
show_results(x,y,pred_gn,cf_gn,5)


print("\nExperiment 1 Plots: Different Initial Estimate\n")

print("a. Gradient Descent")
pred_gd1 = gaussian(x,param_gd1[0],param_gd1[1],param_gd1[2])
show_results(x,y,pred_gd1,cf_gd1,10)
print("b. Gauss Newton")
pred_gn1 = gaussian(x,param_gn1[0],param_gn1[1],param_gn1[2])
show_results(x,y,pred_gn1,cf_gn1,6)


print("\nExperiment 2 Plots: Different Number of Observations\n")
print("a. Gradient Descent")
pred_gd2 = gaussian(nx,param_gd2[0],param_gd2[1],param_gd2[2])
show_results(nx,ny,pred_gd2,cf_gd2,11)
print("b. Gauss Newton")
pred_gn2 = gaussian(nx,param_gn2[0],param_gn2[1],param_gn2[2])
show_results(nx,ny,pred_gn2,cf_gn2,7)


print("\nExperiment 3 Plots: Added Noise\n")

print("a. Gradient Descent")
pred_gd3 = gaussian(x,param_gd3[0],param_gd3[1],param_gd3[2])
show_results(x,noise_y,pred_gd3,cf_gd3,12)
print("b. Gauss Newton")
pred_gn3 = gaussian(x,param_gn3[0],param_gn3[1],param_gn3[2])
show_results(x,noise_y,pred_gn3,cf_gn3,8)