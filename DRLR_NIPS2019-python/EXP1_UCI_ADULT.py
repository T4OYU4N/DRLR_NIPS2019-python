# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:45:14 2023

@author: A
"""
import numpy as np
import LP_ADMM as LA
import Golden_search as gs
import time
#from libsvm.svmutil import svm_read_problem

from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file


kappa = 1
epsilon = 0.1
fid = open('experiment_result/CPUtime_UCI_Adult.txt', 'a+')


for i in range(1, 10):
    dataset_name = 'a' + str(i) + 'a'
    path = 'dataset/' + dataset_name
    print("Processing...dataset:", dataset_name)
    #y_train, x_train = svm_read_problem(path)
    
    x_train, y_train = load_svmlight_file(path)
    x_train_matrix = csr_matrix(x_train).toarray()
    y_train = y_train.reshape(-1,1)
    
    A = np.multiply(x_train_matrix, y_train)
    N, d = A.shape
    Hessian = np.dot(A.T, A)
    AT = A.T
    L = max(np.linalg.eigvals(Hessian))
    LPADMM_param = {
        'd': d,
        'N': N,
        'maxiter': 100000,
        'epsilon': epsilon,
        'delta': 1e-5,
        'rho': 0.001,
        'gamma': 1.04,
        'kappa': kappa,
        'Hessian': Hessian,
        'AT': AT,
        'L': L
    }
    tic = time.time()
    lambda_opt, _ = gs.Golden_search(A, LPADMM_param)
    LPADMM_param['lambda'] = lambda_opt
    LPADMM_param['b'] = lambda_opt * kappa
    LPADMM_output = LA.LP_ADMM(A, LPADMM_param, 0)
    objective = LPADMM_output['objective'] + lambda_opt * LPADMM_param['epsilon']
    Our_cputime = time.time() - tic

    # data = {
    #     'x': x_train.T,
    #     'y': y_train.T
    # }
    # solver_param = {
    #     'kappa': kappa,
    #     'pnorm': np.inf,
    #     'ell': 1,
    #     'epsilon': epsilon,
    #     'solver': 'ipopt'
    # }
    # tic = time.time()
    # solver_output = DRLR(data, solver_param)
    # Solver_cputime = time.time() - tic
    # Solver_obj = solver_output['objective']
    # Solver_lambda = solver_output['lambda']
    mean_our = np.mean(Our_cputime)
    var_our = np.sqrt(np.var(Our_cputime))
    # mean_solver = np.mean(Solver_cputime)
    # var_solver = np.sqrt(np.var(Solver_cputime))
    # Ratio = np.ceil(mean_solver / mean_our)
    fid.write("(N,d,gamma,Rho) = ({},{},{:.2e},{:.0e}), our_cputime:{:.6e}(+-){:.6e};\n".format(
        N, d, LPADMM_param['gamma'], LPADMM_param['rho'], mean_our, var_our))

fid.close()
print("Completed")
print("The results are saved as CPUtime_UCI_Adult.txt")