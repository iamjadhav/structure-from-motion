import numpy as np


def extract_camera_pose(e_mat):
    '''
    Calculating all possible Camera poses, correct if det R = -1
    '''
    u_e, _, v_e = np.linalg.svd(e_mat)
    d_ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    r_1 = np.dot(u_e, np.dot(d_, v_e))
    c_1 = u_e[:, 2]
    #
    r_2 = r_1
    c_2 = -u_e[:, 2]
    #
    r_3 = np.dot(u_e, np.dot(d_.T, v_e))
    c_3 = u_e[:, 2]
    #
    r_4 = r_3
    c_4 = -u_e[:, 2]
    #
    rot_final = [r_1,r_2,r_3,r_4]
    c_final = [c_1,c_2,c_3,c_4]
    ## Important:  det(R) = 1. If det(R) = −1, the camera pose must be corrected i.e. C = −C and R = −R
    for i,curr_r in enumerate(rot_final):
        if np.linalg.det(curr_r) < 0:
            rot_final[i] = -curr_r
            c_final[i] = -c_final[i]
        c_final[i] = c_final[i].reshape((3,1))

    return rot_final,c_final
