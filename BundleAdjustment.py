import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from BuildVisibilityMatrix import build_visibility_matrix


def get_2d_points_camera_indices(X_ids, features_x, features_y, v_mat):
    '''
    2D point correspondences of their 3D counterparts from visibility matrix 
    (present in all views upto current cam_id)
    '''
    pts_2d = []
    point_ids, camera_ids = [], []
    vis_x_feats = features_x[X_ids]
    vis_y_feats = features_y[X_ids]
    for i in range(v_mat.shape[0]): # rows on visibility matrix
        for j in range(v_mat.shape[1]): # columns on visibility matrix
            if v_mat[i,j] == 1:
                pts_2d.append(np.hstack((vis_x_feats[i,j], vis_y_feats[i,j])))
                point_ids.append(i) # rows==matches
                camera_ids.append(j) # columns==images
    #
    points_2d = np.array(pts_2d).reshape(-1, 2)
    # Indices of visible points
    point_ids = np.array(point_ids).reshape(-1)
    # Indices of visible cameras
    camera_ids = np.array(camera_ids).reshape(-1)

    return points_2d, point_ids, camera_ids


def bundle_adjustment_sparsity(n_cam, inlier_X_all_indices, inlier_ids, x_feat_coors, y_feat_coors):
    '''
    Sparsity Matrix
    Referred from: "Large scale BA - SciPy 
    (https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html)"
    '''
    curr_view = n_cam + 1

    X_ids, visibility_matrix = build_visibility_matrix(inlier_X_all_indices, inlier_ids, n_cam)
    n_observations = np.sum(visibility_matrix)
    n_points = len(X_ids[0])
    m = n_observations * 2
    n = curr_view * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(n_observations)
    _, point_indices, camera_indices = get_2d_points_camera_indices(X_ids, x_feat_coors, y_feat_coors, visibility_matrix)
    for s in range(9):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cam * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cam * 6 + point_indices * 3 + s] = 1

    return A


def fun(x0, n_cam, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    Referred from: "Large scale BA - SciPy 
    (https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html)"
    """
    total_cameras = n_cam + 1
    camera_params = x0[:total_cameras * 6].reshape((total_cameras, 6))
    points_3d = x0[total_cameras * 6:].reshape((n_points, 3))
    # points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)

    x_proj_list = []
    points_3d = points_3d[point_indices]
    camera_params = camera_params[camera_indices]

    for i in range(len(camera_params)):
        R_ = Rotation.from_rotvec(camera_params[i, :3])
        R = R_.as_matrix()
        C = camera_params[i, 3:].reshape(3,1)
        point_3d = points_3d[i]
        P = np.dot(K, np.dot(R, np.concatenate((np.identity(3), -C), axis=1)))
        x_4 = np.hstack((point_3d, 1))
        x_projected = np.dot(P, x_4.T)
        x_projected = x_projected/x_projected[-1]
        pt_proj = x_projected[:2]
        x_proj_list.append(pt_proj)
    x_projections = np.array(x_proj_list)
    error_vec = (x_projections - points_2d).ravel()

    return error_vec



def bundle_adjustment(x_feat_coors, y_feat_coors, inlier_ids, inlier_X_all, inlier_X_all_indices, all_rotations, all_translations, K, n_cam):
    '''
    Bundle Adjustment: Refining camera poses and 3D points simultaneously 
    by minimizing the reprojection error
    '''
    ## Get 2D, 3D points, visibility matrix, visible point indices
    X_ids, vis_matrix = build_visibility_matrix(inlier_X_all_indices, inlier_ids, n_cam)
    # print(X_id, vis_matrix)
    # print(X_ids[0].shape, vis_matrix.shape)
    ## Points in all views until n_cam
    points_3d = inlier_X_all[X_ids]
    points_2d, _, _ = get_2d_points_camera_indices(X_ids, x_feat_coors, y_feat_coors, vis_matrix)
    # print(points_3d[:5], points_2d[:5])
    print(points_3d.shape, points_2d.shape)
    ## Get camera params
    RC_params = []
    for i in range(n_cam + 1):
        R, C = all_rotations[i], all_translations[i]
        Q_ = Rotation.from_matrix(R)
        Q = Q_.as_rotvec()
        RC = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC_params.append(RC)
    RC_params = np.array(RC_params).reshape(-1, 6)
    ## Params for least squares
    # Indices of visible cameras and points
    _, point_indices, camera_indices = get_2d_points_camera_indices(X_ids, x_feat_coors, y_feat_coors, vis_matrix)

    x0 = np.concatenate((RC_params.ravel(), points_3d.ravel()), axis=0)
    n_points = points_3d.shape[0]

    A = bundle_adjustment_sparsity(n_cam, inlier_X_all_indices, inlier_ids, x_feat_coors, y_feat_coors)
    # print(A.shape)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cam, n_points, camera_indices, point_indices, points_2d, K))
    t1 = time.time()
    print(f'\nBundle Adjustment for view {n_cam+1} completed in {round(t1-t0, 3)} seconds, A matrix shape {A.shape}')

    x_1 = res.x
    total_cameras = n_cam + 1
    optim_camera_params = x_1[:total_cameras * 6].reshape((total_cameras, 6))
    optimized_3d_points = x_1[total_cameras * 6:].reshape((n_points, 3))

    optim_3d_pts_all = np.zeros_like(inlier_X_all)
    optim_3d_pts_all[X_ids] = optimized_3d_points

    optim_R_set, optim_C_set = [], []
    for i in range(len(optim_camera_params)):
        R_ = Rotation.from_rotvec(optim_camera_params[i, :3])
        R = R_.as_matrix()
        C = optim_camera_params[i, 3:].reshape(3,1)
        optim_R_set.append(R)
        optim_C_set.append(C)

    return optim_R_set, optim_C_set, optim_3d_pts_all