import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


# Camera Calibration matrix for both cameras
K = np.array([[568.996140852, 0, 643.21055941],[0, 568.988362396, 477.982801038],[0, 0, 1]]).reshape(3,3)


def load_images(folder, number_of_images):
    '''
    Loading the stereo images
    '''
    image_list = []
    for i in range(number_of_images):
        for file in glob.glob(folder+'/'+str(i+1)+'.jpg'):
            image = cv2.imread(file)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if image is not None:
                image_list.append(image)
            else:
                print('This image cannot be read.')
    return image_list, K


def draw_features(image_1, image_2, source_feats, destination_feats):
    '''
    To draw the correspondences
    '''
    im_features = np.concatenate((image_1, image_2), axis=1)
    source_feats = source_feats.astype(int)
    destination_feats = destination_feats.astype(int)

    # print(source_feats[1400][0])
    # print(destination_feats[1400][0])
    for i, _ in enumerate(source_feats):
        from_point = tuple((source_feats[i][0], source_feats[i][1]))
        # To compensate for the concatenation (on X-axis), add the X coor of the first image
        # to the X coor of the end_point
        to_point = tuple((destination_feats[i][0] + image_1.shape[1], destination_feats[i][1]))
        cv2.line(im_features, from_point, to_point,
                 color=(0, 255, 0), thickness=1)
    cv2.imshow('image_one_two_all_matches', im_features)
    # cv2.imwrite('./Results/im_1_2_all_matches.jpg', im_features)
    cv2.waitKey(500)
    cv2.destroyAllWindows()


def draw_best_features(image_1, image_2, source_feats, destination_feats, best_matches):
    '''
    To draw the best correspondences between image 0 and image 1
    '''
    im_features = np.concatenate((image_1, image_2), axis=1)
    source_feats = source_feats[best_matches].astype(int)
    destination_feats = destination_feats[best_matches].astype(int)

    # print(source_feats[1400][0])
    # print(destination_feats[1400][0])
    for i, _ in enumerate(source_feats):
        from_point = tuple((source_feats[i][0], source_feats[i][1]))
        # To compensate for the concatenation (on X-axis), add the X coor of the first image
        # to the X coor of the end_point
        to_point = tuple((destination_feats[i][0] + image_1.shape[1], destination_feats[i][1]))
        cv2.line(im_features, from_point, to_point,
                 color=(0, 255, 0), thickness=1)
    cv2.imshow('image_one_plus_two_inliers', im_features)
    # cv2.imwrite('./Results/im_1_2_inliers.jpg', im_features)
    cv2.waitKey(500)
    cv2.destroyAllWindows()


def draw_reprojected_measured_pts(dst_features, points_3d, R, C, K, image, before):
    '''
    Draw measured and reprojected points alongside the ground truth
    Ground Truth: Green
    Triangulated 3D points: Red
    Refined 3D points with NLT: Blue
    '''
    if before is True:
        color = (0,0,255)
    else:
        color = (255,0,0)
    P = np.dot(K, np.dot(R, np.concatenate((np.identity(3), -C), axis=1)))
    # print(points_3d.shape)
    # print(refined_points_3d.shape)
    # print(points_3D.shape)
    reproj_lt = np.dot(P, points_3d.T)
    reproj_lt = reproj_lt/reproj_lt[2]
    reproj_lt = reproj_lt.T

    # reproj_nlt = np.dot(P, refined_points_3d.T)
    # reproj_nlt = reproj_nlt/reproj_nlt[2]
    # reproj_nlt = reproj_nlt.T

    # print(reproj_lt[:5], reproj_lt.shape)
    # print(reproj_lt[3,0])
    # print(reproj_nlt[:5], reproj_nlt.shape)
    #
    for i in range(len(dst_features)):
        cv2.circle(image, (int(dst_features[i,0]), int(dst_features[i,1])), 2, (0,255,0), -1)
        cv2.circle(image, (int(reproj_lt[i,0]), int(reproj_lt[i,1])), 2, color, -1)
    # for i in range(len(dst_features)):
    #     cv2.circle(image, (int(reproj_nlt[i,0]), int(reproj_nlt[i,1])), 2, (0,0,255), -1)
    if before is False:
        cv2.imshow("GT-LT-NLT", image)
        # cv2.imwrite('./Results/GT-LT-NLT.jpg', image)
        cv2.waitKey(50)
        cv2.destroyAllWindows()


def load_data(folder, number_of_images):
    '''
    Loading the point correspondences from the matching.txts
    '''
    descriptors = []
    x_coors = []
    y_coors = []
    coor_flags = []

    for i in range(1, number_of_images):
        try:
            file = open(folder + 'matching' + str(i) + '.txt', 'r')
        except FileNotFoundError:
            print('File cannot be read !')

        for row_num, each_row in enumerate(file):
            if row_num == 0:
                # print(int(each_row.split(':')[1]))
                pass
            else:
                each_line = each_row.split()
                # print(each_line)
                x_arr = np.zeros((1, number_of_images))
                y_arr = np.zeros((1, number_of_images))
                flags_arr = np.zeros((1, number_of_images), dtype=int)
                # Of the form
                # [n_features R G B (u_current image) (v_current image) (image id) (u_image id image) (v_image id image) (image id) (u_image id image) (v_image id image)]
                feature_matches = [float(m) for m in each_line]
                # Total matches
                n_matches = feature_matches[0]
                # RGB values of the correspondence
                red_val = feature_matches[1]
                green_val = feature_matches[2]
                blue_val = feature_matches[3]
                descriptors.append([red_val, green_val, blue_val])
                # Current image feature coordinates
                source_x = feature_matches[4]
                source_y = feature_matches[5]
                # Copy them to the corresponding number in the zero arrays
                x_arr[0][i-1] = source_x
                y_arr[0][i-1] = source_y
                flags_arr[0][i-1] = 1
                # print(x_arr)

                step = 1
                for _ in range(int(n_matches) - 1):
                    # Next match image id and X, Y coors
                    # [n_features R G B (u_current image) (v_current image) (image id) (u_image id image) (v_image id image) (image id) (u_image id image) (v_image id image)]
                    image_id = int(feature_matches[5+step])
                    x_next_im = feature_matches[6+step]
                    y_next_im = feature_matches[7+step]
                    step += 3
                    # Copy them to the corresponding image id e.g. [src_x, 0, next_image_id, 0, next_next_image_id]
                    x_arr[0][image_id-1] = x_next_im
                    y_arr[0][image_id-1] = y_next_im
                    flags_arr[0][image_id-1] = 1

                x_coors.append(x_arr)
                y_coors.append(y_arr)
                coor_flags.append(flags_arr)
    # Only X coordinates of correspondences
    x_coors = np.array(x_coors).reshape(-1, number_of_images)
    # Only Y coordinates of correspondences
    y_coors = np.array(y_coors).reshape(-1, number_of_images)
    coor_flags = np.array(coor_flags).reshape(-1, number_of_images)
    descriptors = np.array(descriptors).reshape(-1, 3)
    # print(x_coors)

    return x_coors, y_coors, coor_flags, descriptors



def plots_2d_3d(X_3d, r_lis, c_lis):
    '''
    Final 2D and 3D plots
    '''

    fig = plt.figure(figsize = (10, 10))

    plt.xlim(-250, 250)
    plt.ylim(-100, 500)

    plt.scatter(X_3d[:,0], X_3d[:,2], marker='.', linewidths=0.5, color = 'blue')

    for i,_ in enumerate((r_lis)):
        euler = Rotation.from_matrix(r_lis[i])
        R1 = euler.as_rotvec()
        R1 = np.rad2deg(R1)
        plt.plot(c_lis[i][0],c_lis[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')

    plt.savefig('./Results/Final_2D.png')
    plt.show()
    plt.pause(5)
    plt.close()

    fig1 = plt.figure(figsize = (10, 10))
    ax1 = plt.axes(projection = "3d")
    ax1.set_title("3D Points")

    ax1.scatter3D(X_3d[:,0], X_3d[:,1], X_3d[:,2], color = "red", s=0.09)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.show()
    plt.pause(5)
    plt.close()
    plt.savefig('./Results/Final_3D.png')
