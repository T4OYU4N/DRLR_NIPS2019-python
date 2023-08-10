# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:02:57 2023
 
@author: A
"""
import numpy as np
import scipy.io as sio
import time
import Golden_search as gs
import LP_ADMM as LA

mat = sio.loadmat('param_sythetic.mat')
Nt = mat['Nt']
D = mat['D']
Gamma = mat['Gamma']
Rho = mat['Rho']

r_fre = 30
Our_cputime = np.zeros(r_fre)
fid = open('experiment_result/CPUtime_Sythetic.txt', 'a+')

for i in range(12):
    d = D[0][i]
    N = Nt[0][i]
    rho = Rho[0][i]
    gamma = Gamma[0][i]
    print("Processing...dataset(N,d)=({}, {})".format(N, d))
    
    for k in range(r_fre):
        np.random.seed(k)
        kappa = 1
        epsilon = 0.1
        beta = np.random.randn(d, 1)
        beta /= np.linalg.norm(beta)
        x = np.random.randn(d, N)
        y = np.array(np.random.rand(1, N) < np.exp(np.dot(beta.T, x)) / (1 + np.exp(np.dot(beta.T, x))), dtype=int)
        y = 2 * y - 1
        A = np.multiply(x, y).T
        
        Hessian = np.dot(A.T, A)
        AT = A.T
        L = np.max(np.linalg.eigvals(Hessian))
        LPADMM_param = {
            'd': d,
            'N': N,
            'maxiter': 100000,
            'epsilon': epsilon,
            'delta': 1e-5,
            'rho': rho,
            'gamma': gamma,
            'kappa': kappa,
            'Hessian': Hessian,
            'AT': AT,
            'L': L
        }
        
        # Our First-order Algorithmic Framework for Synthetic Data

        
        tic = time.time()
        lambda_opt, _ = gs.Golden_search(A, LPADMM_param)
        Our_cputime[k] = time.time() - tic
        LPADMM_param['lambda'] = lambda_opt
        LPADMM_param['b'] = lambda_opt * kappa
        
        LPADMM_output = LA.LP_ADMM(A, LPADMM_param, 0)
        objective = LPADMM_output['objective'] + lambda_opt * LPADMM_param['epsilon']
        
        # # Yamlip Solver
        # solver_param = {
        #     'kappa': kappa,
        #     'pnorm': np.inf,
        #     'ell': 1,
        #     'epsilon': epsilon,
        #     'solver': 'ipopt'
        # }
        
        # tic = time.time()
        # solver_output = DRLR(data, solver_param)
        # Solver_cputime[k] = time.time() - tic
        
        # Solver_obj[k] = solver_output['objective']
        # Solver_lambda[k] = solver_output['lambda']
    
    mean_our = np.mean(Our_cputime)
    var_our = np.sqrt(np.var(Our_cputime))
    # mean_solver = np.mean(Solver_cputime)
    # var_solver = np.sqrt(np.var(Solver_cputime))
    # Ratio = np.ceil(mean_solver / mean_our)
    
    fid.write("(N,d,gamma,Rho) = ({}, {}, {:.2e}, {:.0e}), our_cputime:{:.6e}(+-){:.6e};\n".format(
        N, d, gamma, rho, mean_our, var_our)
    )
fid.close()
    
print("Completed. The results are saved as CPUtime_Sythetic.txt")
