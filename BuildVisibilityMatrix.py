import numpy as np




def build_visibility_matrix(inlier_X_all_indices, inlier_ids, curr_cam_id):
    '''
    Visibility matrix: relationship between a camera and point, construct a I×J
    binary matrix, V where V_ij is one if the j_th point is visible from the i_th
    camera and zero otherwise.
    '''
    temp = np.zeros((inlier_ids.shape[0]), dtype=int)
    # print(temp[:5], temp.shape)
    ## For image columns upto current image i.e. if curr_cam_id =2 -> upto 3rd image
    ## Binary matrix for points that are in any views upto curr_cam_id
    for i in range(curr_cam_id + 1):
        temp = temp | inlier_ids[:,i]
    
    three_d_inliers = inlier_X_all_indices.reshape(-1)
    # Indices of the 3D inliers present in views upto curr_cam_id
    X_ids = np.where(three_d_inliers & temp)
    # Indices of those inlier pts
    vis_matrix = inlier_X_all_indices[X_ids].reshape(-1,1)

    for i in range(curr_cam_id + 1):
        vis_matrix = np.concatenate((vis_matrix, inlier_ids[X_ids, i].reshape(-1,1)), axis=1)
    
    _, w = vis_matrix.shape
    vis_matrix = vis_matrix[:, 1:w]

    return X_ids, vis_matrix
