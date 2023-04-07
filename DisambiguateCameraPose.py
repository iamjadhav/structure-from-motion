import numpy as np
import matplotlib.pyplot as plt
from LinearTriangulation import linear_triangulation


def disambiguate_camera_pose(source_feats, dest_feats, r_all, c_all, K, visualize):
    '''
    Perform Linear Triangulation to get reconstructed 3D points and disambiguate
    to get the depth-positive camera pose
    '''
    # print('\n')
    max_posdepth_pts = 0
    r_final, c_final, final_3d_points = None, None, []
    r_1 = np.identity(3)
    c_1 = np.zeros((3,1))

    for i,_ in enumerate(r_all):
        three_d_points = linear_triangulation(r_1, c_1, r_all[i], c_all[i], source_feats, dest_feats, K)
        three_d_points = three_d_points/three_d_points[:,3].reshape(-1,1)
        # print('dis 1: ', three_d_points[:5], three_d_points.shape) #homogeneous coordinates
        # print(r_all[i])
        # print('1', r_all[i][-1])
        # print('2', r_all[i][2])
        ##### r3 (X - C) > 0 : positive depth condition #####
        # Last row of R is r3
        r3 = r_all[i][-1].reshape(1,-1)
        # print('r3', r3, r3.shape)
        # print(three_d_points[i])
        # print(three_d_points[i]/three_d_points[i][-1])
        # X = three_d_points[i]/three_d_points[i][-1].reshape(-1,1)
        # print(X)
        # X = X[:,:-1]
        # print(X)
        # print('X upto 3rd',X.shape)
        C = c_all[i]
        # print(C.shape)
        three_d_points = three_d_points[:, :3]
        # print('dis 2: ', three_d_points[:5], three_d_points.shape)
        chierality_condition = np.dot(r3, (three_d_points.T - C)).reshape(-1,1)
        # print('chier cond ', chierality_condition[:5], chierality_condition.shape)
        # print(chierality_condition.T)
        # print(len(np.where(chierality_condition > 0)[0]))
        chierality_condition = chierality_condition > 0
        ##### z > 0: This will be almost all Trues for the correct unique camera pose #####
        positive_depth_condition = three_d_points[:,2] > 0
        # print(positive_depth_condition)
        points_in_front = chierality_condition * positive_depth_condition
        total_pos_pts = np.sum(points_in_front)

        if total_pos_pts > max_posdepth_pts:
            max_posdepth_pts = total_pos_pts
            r_final = r_all[i]
            c_final = c_all[i]
            final_3d_points = three_d_points
        else:
            # print(total_pos_pts)
            continue

    if visualize:
        # Linear_triangulation.png plot for 3D points calculated using the
        # disambiguated pose
        plt.scatter(final_3d_points[:,0], final_3d_points[:,2], s=0.09)
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Linear Triangulation')
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    return r_final, c_final, final_3d_points
