import numpy as np
from EstimateFundamentalMatrix import estimate_fundamental_matrix



def inliers_ransac(source_feats, dest_feats):
    '''
    To get inliers in all possible image pairs
    ''' 
    N = 2000
    sample = 0
    thresh = 0.5
    # e = 0.5
    selected_f_mat = None
    best_match_ids = None
    maximum_inliers = 0
    while N > sample:

        src_list, dst_list = [], []
        current_inliers = 0

        rand_ids = np.random.choice(len(source_feats), size=8)
        # print(rand_ids)
        for i in rand_ids:
            src_list.append(source_feats[i])
            dst_list.append(dest_feats[i])
        # print(src_list)

        f_mat = estimate_fundamental_matrix(src_list, dst_list)
        # print(f_mat)
        # Equation
        ones_column = np.ones((len(source_feats),1))
        x_1 = np.concatenate((source_feats, ones_column), axis=1)
        y_1 = np.concatenate((dest_feats, ones_column), axis=1)
        # x_2.T * F * x_1
        err = y_1@f_mat
        e_1 = np.sum((err * x_1), axis=1)
        e_2 = ((np.sum(err[:,:2] ** 2, axis=1)) ** 0.5)
        error = e_1 / e_2
        error = np.abs(error)

        inliers = error <= thresh
        current_inliers = np.sum(inliers)
        # Estimating best Fundamental Matrix
        if maximum_inliers < current_inliers:
            maximum_inliers = current_inliers
            selected_f_mat = f_mat
            best_match_ids = inliers

        sample+=1

    return selected_f_mat, best_match_ids
