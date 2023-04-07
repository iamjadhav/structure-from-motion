import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation


def non_linear_pnp(R, C, dest_feats, X_curr_im, K):
    '''
    Non-linear PnP
    '''
    # A compact representation of the rotation matrix using quaternion is a better choice
    # to enforce orthogonality of the rotation matrix, R = R(q) where q is four dimensional quaternion
    if R is not 0 and C is not 0:
        Q_ = Rotation.from_matrix(R)
        Q = Q_.as_quat()
        # print(Q)
        QC = [Q[0], Q[1], Q[2], Q[3], C[0], C[1], C[2]]
        # print(QC)
        optim_result = scipy.optimize.least_squares(fun=loss_non_linear_pnp, method='trf', x0=QC, args=[X_curr_im, dest_feats, K])
        X = optim_result.x
        optim_q = X[:4]
        C = X[4:]
        R_ = Rotation.from_quat(optim_q)
        R = R_.as_matrix()
        return R, C
    else:
        return 0, 0



def loss_non_linear_pnp(X, X_3d, pts_2d, K):
    '''
    Non-linear PnP loss
    '''
    X_3d = np.column_stack((X_3d, np.ones(len(X_3d))))
    Q = X[:4]
    C = X[4:].reshape(-1,1)
    ## R matrix orthogonality enforced using 4D quaternion: R=R(q)
    R_ = Rotation.from_quat(Q)
    R = R_.as_matrix()
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
