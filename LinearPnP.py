import numpy as np



def linear_pnp(image_pts_x, threeD_pts, K):
    '''
    Linear PnP to uniquely identify the camera poses of successive views
    '''
    # 6 Homogeneous image points
    x_tilda = np.concatenate((image_pts_x, np.ones(len(image_pts_x)).reshape(-1,1)), axis=1) #6X1
    # 6 Homogeneous 3D points
    X_tilda = np.concatenate((threeD_pts, np.ones(len(threeD_pts)).reshape(-1,1)), axis=1) #6X1
    # print('x homo ', x_tilda)
    # print('X homo ', X_tilda)
    #### Nomralizing homogeneous correspondences ####
    x_norm_pts = np.dot(np.linalg.inv(K), x_tilda.T).T #6X1
    # print('x norm', x_norm_pts)
    A_mat = []
    for i in range(len(threeD_pts)): #6
        zero_column = np.zeros((1,4))
        X = X_tilda[i].reshape((1,4))
        u, v, _ = x_norm_pts[i]

        uv_arr = np.array([[0, -1, v],
                           [1, 0, -u],
                           [-v, u, 0]])
        # print(uv_arr.shape)
        row_1 = np.concatenate((X, zero_column, zero_column), axis=1)
        row_2 = np.concatenate((zero_column, X, zero_column), axis=1)
        row_3 = np.concatenate((zero_column, zero_column, X), axis=1)

        X_homo_arr = np.concatenate((row_1, row_2, row_3), axis=0)
        # print('X homo arr ', X_homo_arr, X_homo_arr.shape)
        curr_pt_a = uv_arr.dot(X_homo_arr)
        if i > 0:
            A_mat = np.vstack((A_mat, curr_pt_a))
        else:
            A_mat = curr_pt_a
    A_mat = np.array(A_mat)
    # print('Final A mat ', A_mat, A_mat.shape)
    _, _, vt = np.linalg.svd(A_mat)
    # print(vt.shape)
    # print(vt[:,-1])
    projection_matrix = vt[:][-1]
    projection_matrix = projection_matrix.reshape(3,4)
    # print(projection_matrix)
    ##### Rotation #####
    rotation_ = projection_matrix[:,:3]
    # print('r init ', rotation_)
    u, _, vT = np.linalg.svd(rotation_)
    # R = U.V_T
    ## Enforcing orthogonality of R using corrected R as U.V_T
    ## and t = -R_inv.C
    rotation_final = u.dot(vT)
    ##### Translation #####
    C = projection_matrix[:,-1]
    # print(C)
    translation_final = -np.dot(np.linalg.inv(rotation_final), C)

    if np.linalg.det(rotation_final) < 0:
        translation_final = -translation_final
        rotation_final = -rotation_final
   
    return rotation_final, translation_final