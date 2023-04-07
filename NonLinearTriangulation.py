import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import cv2


def nonlinear_triangulation(source_feats, dest_feats, threeD_points, R1, C1, R_final, C_final, K, visualize):
    '''
    To minimize the reprojection error between the calculated and projected 3D points
    '''
    # print(C_final, C_final.shape)
    mean_error = False
    P1 = np.dot(K, np.dot(R1, np.concatenate((np.identity(3), -C1), axis=1)))
    P2 = np.dot(K, np.dot(R_final, np.concatenate((np.identity(3), -C_final), axis=1)))
    # print(P1, P2)
    # print(pts1, pts1.shape)
    # print(threeD_points[:5])
    # threeD_points = np.column_stack((threeD_points, np.ones(len(threeD_points))))

    optimized_3d_pts = []
    for i,_ in enumerate(source_feats):
        optim_result = scipy.optimize.least_squares(fun=error_function, x0=threeD_points[i], method="trf", 
                                                    args=[source_feats[i], dest_feats[i], P1, P2, mean_error])
        X = optim_result.x
        optimized_3d_pts.append(X)

    refined_threeD_points = np.array(optimized_3d_pts)
    # print(threeD_points[:5])

    if visualize:
        # Non-linear_triangulation.png plot for refined 3D points calculated using the
        # scipy least square optimization
        plt.scatter(threeD_points[:,0], threeD_points[:,2], color='blue', s=0.09)
        plt.scatter(refined_threeD_points[:,0], refined_threeD_points[:,2], color='red', s=0.09)
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Non-linear Triangulation')
        # plt.show()
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    return refined_threeD_points


def error_function(X_3d, src_points, dst_points, P1, P2, mean_error):
    '''
    The loss function for NLT to minimize reprojection error to get optimal X points
    ''' 
    p1_col1, p1_col2, p1_col3 = P1
    p1_col1, p1_col2, p1_col3 = p1_col1.reshape(1,-1), p1_col2.reshape(1,-1),p1_col3.reshape(1,-1)

    p2_col1, p2_col2, p2_col3 = P2
    p2_col1, p2_col2, p2_col3 = p2_col1.reshape(1,-1), p2_col2.reshape(1,-1), p2_col3.reshape(1,-1)

    # Reprojection error for reference camera points - j = 1
    u_1 = src_points[0]
    v_1 = src_points[1]
    u_1_proj = np.divide(p1_col1.dot(X_3d), p1_col3.dot(X_3d))
    v_1_proj =  np.divide(p1_col2.dot(X_3d), p1_col3.dot(X_3d))
    e_1 = np.square(v_1 - v_1_proj) + np.square(u_1 - u_1_proj)
    # Reprojection error for second camera points - j = 2
    u_2 = dst_points[0]
    v_2 = dst_points[1]
    u_2_proj = np.divide(p2_col1.dot(X_3d), p2_col3.dot(X_3d))
    v_2_proj =  np.divide(p2_col2.dot(X_3d), p2_col3.dot(X_3d)) 
    e_2 = np.square(v_2 - v_2_proj) + np.square(u_2 - u_2_proj)
    if mean_error:
        return e_1, e_2
    else:
        pass
    error = e_1 + e_2
    error = error.squeeze()
    return error


def mean_repro_error(threeD_points, src_points, dst_points, R1, C1, R_final, C_final, K):
    '''
    Mean reprojection error calculation for error before and after NLT
    '''
    mean_error = True
    P1 = np.dot(K, np.dot(R1, np.concatenate((np.identity(3), -C1), axis=1)))
    P2 = np.dot(K, np.dot(R_final, np.concatenate((np.identity(3), -C_final), axis=1)))

    error = []
    for src_pt, dest_pt, X in zip(src_points, dst_points, threeD_points):
        e_1, e_2 = error_function(X, src_pt, dest_pt, P1, P2, mean_error)
        error.append(e_1 + e_2)
    mean_e = np.mean(error)
    return mean_e
