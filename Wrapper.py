'''
Structrure-From-Motion: Pipeline to reconstruct a 3D scene from 6 stereo images.
'''
import time
import pdb
import numpy as np
np.set_printoptions(precision=3)
from LoadData import load_images, load_data, draw_features, draw_best_features, draw_reprojected_measured_pts, plots_2d_3d
from GetInliersRANSAC import inliers_ransac
from EstimateFundamentalMatrix import estimate_fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import estimate_essential_matrix
from ExtractCameraPose import extract_camera_pose
from DisambiguateCameraPose import disambiguate_camera_pose
from LinearTriangulation import linear_triangulation
from NonLinearTriangulation import nonlinear_triangulation, error_function, mean_repro_error
from PnPRANSAC import pnp_ransac, error_reprojection_pnp
from NonLinearPnP import non_linear_pnp
from BundleAdjustment import bundle_adjustment

# Camera calibration matrix
# K = np.array([[568.996140852, 0, 643.21055941],[0, 568.988362396, 477.982801038],[0, 0, 1]])


def main():
    '''
    The entire pipeline strung together
    '''
    visualize = False
    data_folder = "Data/"
    total_images = 6
    # Load dataset
    images, K = load_images(data_folder, total_images)
    # X, Y coors, Flags, Descriptors with zeros
    x_feat_coors, y_feat_coors, feature_indices, descriptors = load_data(data_folder, total_images)
    # Shape (10331,6)
    # print(x_feat_coors[:5], x_feat_coors.shape) # X's for all images having the same match
    # print(y_feat_coors[:5], y_feat_coors.shape) # Y's for all images having the same match
    # print(feature_indices[:5], feature_indices.shape) # Indices of the matches

    ####################### Display Features using Image 1 and 2 #######################
    if visualize:
        # All matching correspondences
        common_ids = np.where(feature_indices[:,0] & feature_indices[:,1]) # bitwise and to find common ids
        # Feature coors for only the respective images i.e. 1 and 2 in this case, e.g [454.54 392.99]
        # Image 1
        src_feats_initial = np.concatenate((x_feat_coors[common_ids,0].reshape(-1,1),
                                       y_feat_coors[common_ids,0].reshape(-1,1)), axis=1)
        # Image 2
        dst_feats_initial = np.concatenate((x_feat_coors[common_ids,1].reshape(-1,1),
                                        y_feat_coors[common_ids,1].reshape(-1,1)), axis=1)
        # print('\n[X Y]s of the 1st (source) image: ', src_feats_initial)
        # print('\n[X Y]s of the 2nd (dest) image: ', dst_feats_initial)
        # print(len(images))
        draw_features(images[0], images[1], src_feats_initial, dst_feats_initial)


    ####################### Rejecting outliers for all possible image pairs #######################
    start_time_1 = time.time()
    inlier_ids = np.zeros_like(feature_indices)
    f_matrix, final_inlier_ids = None, None
    for i in range(0, total_images - 1): # source image
        for j in range(i+1, total_images): # dest image
            common_ids_ransac = np.where(feature_indices[:,i] & feature_indices[:,j]) ## All matches between images i and j (Tuple)
            # print(common_ids_ransac[0][:5], common_ids_ransac[0].shape)
            # print("\nTotal correpondences: ",len(common_ids_ransac[0]))
            source_feats_ransac = np.concatenate((x_feat_coors[common_ids_ransac,i].reshape((-1,1)),
                                           y_feat_coors[common_ids_ransac,i].reshape((-1,1))), axis=1)
            dest_feats_ransac = np.concatenate((x_feat_coors[common_ids_ransac,j].reshape((-1,1)),
                                           y_feat_coors[common_ids_ransac,j].reshape((-1,1))), axis=1)
            # print(source_feats_ransac[:5], dest_feats_ransac[:5])
            all_matches = common_ids_ransac ## All matches between images i and j (Tuple)
            common_ids_ransac = np.array(common_ids_ransac).reshape(-1)
            # print(len(common_ids_ransac))
            
            if len(common_ids_ransac) > 8:
                # Best F mat, Inlier boolean array with True value for inlier indices, False on outliers
                f_matrix, inliers = inliers_ransac(np.array(source_feats_ransac), np.array(dest_feats_ransac))
                # print(f_matrix.shape)
                # print(inliers[:10], inliers.shape)
                # print(all_matches[0][:10], all_matches[0].shape)
                ## Final row numbers of the inliers
                final_inlier_ids = all_matches[0][inliers]
                # print('final inlier ids ', final_inlier_ids[:10], final_inlier_ids.shape)
                ## Make the flags for the inlier rows equal 1 for columns i and j (for those images)
                inlier_ids[final_inlier_ids, i] = 1
                inlier_ids[final_inlier_ids, j] = 1
                # print(inlier_ids[:10], inlier_ids.shape)
                print(f'{len(all_matches[0])} matches and {len(final_inlier_ids)} inliers between image {i} and image {j}')
                if i < j <= 1 and visualize:
                    draw_best_features(images[0], images[1], src_feats_initial, dst_feats_initial, inliers)
            else:
                # print(f'Not enough matches {len(all_matches[0])} between image {i} and image {j}')
                continue
    time_to_find_inliers = time.time() - start_time_1
    print(f'\nTook {round(time_to_find_inliers,2)} seconds to find the inliers')

    # <<<<<<<<<<<<<-------------------- Sequantial Registration Method -------------------->>>>>>>>>>>>>
    # They work by incorporating successive views one at a time.
    # A suitable initialization is typically obtained by decomposing the fundamental matrix relating
    # the first two views of the sequence.
    ####################### Essential matrix from F matrix #######################
    source_image = images[0].copy()
    destination_image = images[1].copy()
    common_ids = np.where(inlier_ids[:,0] & inlier_ids[:,1])
    src_feats_initial = np.concatenate((x_feat_coors[common_ids,0].reshape(-1,1),
                                   y_feat_coors[common_ids,0].reshape(-1,1)), axis=1)
    dst_feats_initial = np.concatenate((x_feat_coors[common_ids,1].reshape(-1,1),
                                   y_feat_coors[common_ids,1].reshape(-1,1)), axis=1)

    f_mat = estimate_fundamental_matrix(src_feats_initial, dst_feats_initial)
    print('\nF Matrix: ')
    print(f_mat)

    e_matrix = estimate_essential_matrix(f_mat, K)
    print('\nE Matrix: ')
    print(e_matrix)
    ####################### All Camera configs #######################
    rotations, translations = extract_camera_pose(e_matrix)
    # print('\nRotations and Translations')
    # print(rotations, '\t', translations)

    ####################### Disambiguate camera poses using Cheirality condition #######################
    ## Using Image One (0) and Two (1)
    r_disam_init, t_disam_init, points_3d = disambiguate_camera_pose(src_feats_initial, dst_feats_initial, rotations, translations, K, visualize)
    points_3d = np.column_stack((points_3d, np.ones(len(points_3d)))) # Already divided by z in triangulation, just adding the 1s
    print('\nFinal Rotation & Translation:')
    print(r_disam_init,'\n', t_disam_init)
    print('\nTriangulated 3D points (Image 1 and 2): ')
    print(points_3d[0:5], points_3d.shape)

    ####################### Minimizing reprojection error using NLT #######################
    ## Using Image One (0) and Two (1)
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    points_3d_nlt = nonlinear_triangulation(src_feats_initial, dst_feats_initial, points_3d, R1, C1, r_disam_init, t_disam_init, K, visualize)
    # Already divided by z in NLT, just adding the 1s
    points_3d_nlt = points_3d_nlt/points_3d_nlt[:,3].reshape(-1,1)
    print('\nRefined 3D points after NLT (Image 1 and 2): ')
    print(points_3d_nlt[:5], points_3d_nlt.shape)

    # Error before Non-linear optimization
    mean_e_before = mean_repro_error(points_3d, src_feats_initial, dst_feats_initial, R1, C1, r_disam_init, t_disam_init, K)
    print('\nMean reprojection error before Non-linear Triangulation: ', round(mean_e_before, 3))
    mean_e_after = mean_repro_error(points_3d_nlt, src_feats_initial, dst_feats_initial, R1, C1, r_disam_init, t_disam_init, K)
    print('Mean reprojection error after Non-linear Triangulation: ', round(mean_e_after, 3))
    # print(points_3d_nlt[:5], points_3d_nlt.shape)

    ## Visualization of measured(triangulated) and reprojected 3D points
    if visualize:
        draw_reprojected_measured_pts(dst_feats_initial, points_3d, r_disam_init, t_disam_init, K, destination_image, before=True)
        draw_reprojected_measured_pts(dst_feats_initial, points_3d_nlt, r_disam_init, t_disam_init, K, destination_image, before=False)

    ####################### Linear PnP, PnP RANSAC, Non-linear PnP #######################
    # With the 3D points in the world, their 2D projections in the image, and K; 
    # The 6 DOF camera pose can be estimated using linear least squares

    all_rotations, all_translations = [], []

    ############# Poses of Camera 1 and 2 #############
    all_rotations.append(R1)
    all_translations.append(C1)
    all_rotations.append(r_disam_init)
    all_translations.append(t_disam_init)

    inlier_3d_all_im = np.zeros((x_feat_coors.shape[0], 3)) # For storing all inlier 3D points
    inlier_3d_all_im_indices = np.zeros((x_feat_coors.shape[0], 1), dtype=int)
    #
    inlier_3d_all_im[common_ids] = points_3d[:, :3] # All inlier 3D points from Image 1 and 2
    inlier_3d_all_im_indices[common_ids] = 1 # Image 1 ids == 1
    # print('inliers image 1: ', inlier_3d_all_im[:5], inlier_3d_all_im.shape)
    # print('inliers image 1 ids: ', inlier_3d_all_im_indices[:5], inlier_3d_all_im_indices.shape)

    # Only +ve depth pts
    inlier_3d_all_im_indices[inlier_3d_all_im[:, 2] < 0] = 0
    # print('pos depth: ', inlier_3d_all_im_indices[:5], inlier_3d_all_im_indices.shape)

    for i in range(2, total_images):
        # Common point ids between first image and the successive images
        c_ids_pnp = np.where(inlier_ids[:,i] & inlier_3d_all_im_indices[:,0])
        print(f'\n{len(c_ids_pnp[0])} common points between image 1 and image {i}')
        # X-Y Correspondences from successive images
        curr_dest_feats_pnp = np.concatenate((x_feat_coors[c_ids_pnp, i].reshape(-1,1),
                                              y_feat_coors[c_ids_pnp, i].reshape(-1,1)), axis=1)
        # Inlier 3D points common between current image and image 1
        X_curr_pnp = inlier_3d_all_im[c_ids_pnp, :].reshape(-1,3)
        # print('Common 3D pts: ', X_curr_pnp[:5], X_curr_pnp.shape)
        ## Calibrated camear pose estimation using 3D-2D correspondences
        r_init_pnp, c_init_pnp = pnp_ransac(curr_dest_feats_pnp, X_curr_pnp, K)
        print('Linear PnP R & C:')
        print(r_init_pnp, '\t', c_init_pnp)
        error_pnp = error_reprojection_pnp(r_init_pnp, c_init_pnp, curr_dest_feats_pnp, X_curr_pnp, K)
        ## Minimize MRE over r_init_pnp, c_init_pnp
        r_pnp_final, c_pnp_final = non_linear_pnp(r_init_pnp, c_init_pnp, curr_dest_feats_pnp, X_curr_pnp, K)
        print('Non-linear PnP R & C:')
        print(r_pnp_final, '\t', c_pnp_final)
        c_pnp_final = np.reshape(c_pnp_final,(3,1))
        #
        error_nlpnp = error_reprojection_pnp(r_pnp_final, c_pnp_final, curr_dest_feats_pnp, X_curr_pnp, K)
        print('MRE before Non-linear PnP:', round(error_pnp,3))
        print('MRE after Non-linear PnP:', round(error_nlpnp,3))
        all_rotations.append(r_pnp_final)
        all_translations.append(c_pnp_final)
        # Adding 3D points of successive views using Triangulation
        for j in range(0, i):
            common_indices = np.where(inlier_ids[:, i] & inlier_ids[:, j])
            # print(len(common_indices[0]))
            if len(common_indices[0]) < 8:
                print('less than 8 pts')
                continue
            # print('Enough matches let go')
            s_features = np.concatenate((x_feat_coors[common_indices, j].reshape(-1,1),
                                         y_feat_coors[common_indices, j].reshape(-1,1)), axis=1)
            d_features = np.concatenate((x_feat_coors[common_indices, i].reshape(-1,1),
                                         y_feat_coors[common_indices, i].reshape(-1,1)), axis=1)
            # print(all_rotations[j].shape, all_translations[j].shape, r_pnp_final.shape, c_pnp_final.shape)
            X_3d = linear_triangulation(all_rotations[j], all_translations[j], r_pnp_final, c_pnp_final, s_features, d_features, K)
            X_3d = X_3d/X_3d[:,3].reshape(-1,1)
            #
            mean_e_before = mean_repro_error(X_3d, s_features, d_features, all_rotations[j], all_translations[j], r_pnp_final, c_pnp_final, K)
            #
            X_3d_nlt = nonlinear_triangulation(s_features, d_features, X_3d, all_rotations[j], all_translations[j], r_pnp_final, c_pnp_final, K, visualize)
            X_3d_nlt = X_3d_nlt/X_3d_nlt[:,3].reshape(-1,1) # Already divided by z in NLT, just adding the 1s
            #
            mean_e_after = mean_repro_error(X_3d_nlt, s_features, d_features, all_rotations[j], all_translations[j], r_pnp_final, c_pnp_final, K)
            print('\nMRE before Non-linear Triangulation: ', round(mean_e_before, 3))
            print('MRE before Non-linear Triangulation: ', round(mean_e_after, 3))
            #
            inlier_3d_all_im[common_indices] = X_3d_nlt[:,:3]
            inlier_3d_all_im_indices[common_indices] = 1
        
        R_set, C_set, final_X = bundle_adjustment(x_feat_coors, y_feat_coors, inlier_ids, inlier_3d_all_im, inlier_3d_all_im_indices,
                                                  all_rotations, all_translations, K, n_cam=i)

    # print(R_set, C_set, final_X)
    print(len(R_set), len(C_set), final_X.shape)
    inlier_3d_all_im_indices[inlier_3d_all_im[:,2] < 0] = 0

    indices = np.where(inlier_3d_all_im_indices[:, 0])
    X = inlier_3d_all_im[indices]

    plots_2d_3d(X, R_set, C_set)
    return




if __name__ == '__main__':
    main()
