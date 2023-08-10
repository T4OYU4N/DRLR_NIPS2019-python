# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:27:03 2023
 
@author: A
"""
import numpy as np
import ACG as acg
import prox_l1 as l1

def LP_ADMM(A, param, test):
    d = param['d']
    N = param['N']
    maxiter = param['maxiter']
    delta = param['delta']  # stopping tolerance
    rho = param['rho']  # initial penalty parameter
    lambd = param['lambda']
    gamma = param['gamma']
    b = param['lambda'] * param['kappa']
    Hessian = param['Hessian']
    AT = param['AT']
    L = param['L']

    # Initialization
    x = np.ones((d, 1))
    y = np.ones((N, 1))
    w = np.ones((N, 1))
    iter = 1

    if test == 0:
        while True:
            iter += 1
            # Adaptive Linearized Proximal ADMM main iterative steps
            rho = gamma * rho
            rho_r = 1 / rho
            x_mid = y + rho_r * w
            ATb = np.dot(AT, x_mid)

            # The quadratic programming with box constraints subproblem solver
            x = acg.ACG(ATb, Hessian, lambd, x)  # Assuming ACG is implemented separately
            
            G = np.dot(A, x)
            #print(min(y))
            
            #grad_f = -np.exp(-y) / (1 + np.exp(-y)) + 0.5
            grad_f = -1/(np.exp(y) + 1) + 0.5
            y_mid = G - rho_r * (w + grad_f / N)
            
            y = l1.prox_l1(y_mid - b, 0.5 * rho_r / N) + b  # Assuming prox_l1 is implemented separately
            u = G - y  # residual
            w = w - rho * u
            max_tol = np.max(np.abs(u))

            if max_tol < delta:
                break
            if iter > maxiter:
                break
        
        output = {
            'beta': x,
            'iter': iter,
            'objective': DRO_obj(A, x, b, N)
        }
    else:
        obj = [DRO_obj(A, x, b, N)]
        time = [0]

        while True:
            tic = time.time()
            iter += 1
            # Adaptive Linearized Proximal ADMM main iterative steps
            rho = gamma * rho
            rho_r = 1 / rho
            x_mid = y + rho_r * w
            ATb = np.dot(AT, x_mid)

            # The quadratic programming with box constraints subproblem solver
            x = acg.ACG(ATb, Hessian, lambd, x)  # Assuming ACG is implemented separately

            G = np.dot(A, x)
            grad_f = -np.exp(-y) / (1 + np.exp(-y)) + 0.5
            y_mid = G - rho_r * (w + grad_f / N)
            y = l1.prox_l1(y_mid - b, 0.5 * rho_r / N) + b  # Assuming prox_l1 is implemented separately
            u = G - y
            w = w - rho * u
            max_tol = np.max(np.abs(u))
            itertime = time.time() - tic
            obj.append(DRO_obj(A, x, b, N))
            print('iter: %d, objective: %1.6e' % (iter, obj[iter]))
            time.append(time[iter - 1] + itertime)

            if max_tol < delta:
                break
            if iter > maxiter:
                break
        
        output = {
            'beta': x,
            'time': time,
            'obj': obj,
            'iter': iter,
            'objective': DRO_obj(A, x, b, N)
        }
    
    return output

def DRO_obj(A, x, b, N):
    objective = np.sum(np.log(1 + np.exp(-np.dot(A, x))) + np.maximum(np.dot(A, x) - b, 0)) / N
    return objective