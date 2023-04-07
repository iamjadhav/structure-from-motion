import numpy as np
from LinearPnP import linear_pnp


def pnp_error(r_init, c_init, dest_feats, X_curr_im, K, thresh):
    '''
    PnP error function to estimate 6DOF camera pose  using 3D-2D pts 
    then using RANSAC to get R,C with most inliers using MRE error function
    '''
    # print(r_init, r_init.shape)
    # print(c_init, c_init.shape)
    inlier_ids = []
    X_ = np.concatenate((X_curr_im, np.ones((len(X_curr_im),1))), axis=1)
    # print(X_)
    P = np.dot(K, np.dot(r_init, np.concatenate((np.identity(3), -c_init), axis=1)))
    p_col1, p_col2, p_col3 = P
    # print(P)
    for i,_ in enumerate(X_curr_im):
        u, v = dest_feats[i]

        u_1_proj = np.divide(p_col1.dot(X_[i]), p_col3.dot(X_[i]))
        v_1_proj = np.divide(p_col2.dot(X_[i]), p_col3.dot(X_[i]))
        # print(u_1_proj, v_1_proj)

        # e_1 = np.square(u - u_1_proj) + np.square(v - v_1_proj) # Too big of an error
        # print(e_1)

        x_proj = np.concatenate(([u_1_proj], [v_1_proj]), axis=0)
        x = np.concatenate(([u], [v]), axis=0)
        e_1 = np.linalg.norm(x - x_proj)
        # print(e_1)
        if e_1 <= thresh:
            inlier_ids.append(i)
    # print(inlier_ids)
    return inlier_ids


def pnp_ransac(dest_feats, X_curr_im, K):
    '''
    PnP RANSAC to get inliers from Linear PnP points
    '''
    N = 5000
    sample = 0
    thresh = 5
    r_pnp, c_pnp = None, None
    # e = 0.5
    maximum_inliers = 0
    while N > sample:

        x_lis, X_3d_lis = [], []
        current_inliers = 0
        if len(dest_feats) != 0:
            rand_ids = np.random.choice(len(dest_feats), size=6)
        else:
            print('\nNo matching points: PnP')
            return 0, 0
        # print(rand_ids)
        for i in rand_ids:
            x_lis.append(dest_feats[i])
            X_3d_lis.append(X_curr_im[i])
        # print(np.array(x_lis))
        # print(np.array(X_3d_lis))

        r_init, c_init = linear_pnp(np.array(x_lis), np.array(X_3d_lis), K)
        # print(r_init, c_init)
        inlier_ids = pnp_error(r_init, c_init.reshape(-1,1), dest_feats, X_curr_im, K, thresh)
        #
        # current_inliers = np.sum(inlier_ids)
        # Estimating best Fundamental Matrix
        if maximum_inliers < len(inlier_ids):
            maximum_inliers = len(inlier_ids)
            r_pnp = r_init
            c_pnp = c_init

        sample+=1

    return r_pnp, c_pnp


def error_reprojection_pnp(R, C, pts_2d, X_3d, K):
    '''
    The reprojection loss function for PnP
    '''
    if R is not 0 and C is not 0:
        X_3d = np.column_stack((X_3d, np.ones(len(X_3d))))
        C = C.reshape((3,1))
        P = np.dot(K, np.dot(R, np.concatenate((np.identity(3), -C), axis=1)))
        error_list = []
        for pt_2d, pt_3d in zip(pts_2d, X_3d):
            #
            p_col1, p_col2, p_col3 = P
            p_col1, p_col2, p_col3 = p_col1.reshape(1,-1), p_col2.reshape(1,-1),p_col3.reshape(1,-1)
            pt_3d = pt_3d.reshape(-1,1)
            u_1, v_1 = pt_2d[0], pt_2d[1]
            u_1_proj = np.divide(p_col1.dot(pt_3d), p_col3.dot(pt_3d))
            v_1_proj =  np.divide(p_col2.dot(pt_3d), p_col3.dot(pt_3d))
            #
            e_1 = np.square(v_1 - v_1_proj) + np.square(u_1 - u_1_proj)       
            error_list.append(e_1)
        error = np.array(error_list)
        error = error.squeeze()
        mean_error = np.mean(error)
        return mean_error
    else:
        return 0

