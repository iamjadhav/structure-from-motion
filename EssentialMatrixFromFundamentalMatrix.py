import numpy as np


def estimate_essential_matrix(f_mat, K):
    '''
    To estimate the E matrix using the F matrix
    '''
    e_mat = K.T.dot(f_mat).dot(K)
    # print(e_mat)
    u_, sigma, v_t = np.linalg.svd(e_mat)
    # print(sigma)
    sigma = np.diag([1, 1, 0])
    e_final = np.dot(u_, np.dot(sigma, v_t))
    # print(e_final)
    return e_final
