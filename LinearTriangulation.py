import numpy as np


def linear_triangulation(R1, C1, R2, C2, src_points, dst_points, K):
    '''
    To reconstruct the 3D points using triangulation: https://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
    Because the homogenous 3-vectors u_{i} and P_{i}X are parallel, it is possible to write:
    [u_{i}]× P_{i} X = 0.
    x = alpha(scalar) . P . X
    [x y z] = alpha . [Projection matrix] . [X Y Z 1]
    [x y z] = alpha [p1T.X p2T.X p3T.X]
    [x y z] cross [p1T.X p2T.X p3T.X] = [0 0 0] -> x X PX = 0
    '''
    # print('\n')
    # C1 = np.reshape(C1,(3,1))
    C2 = np.reshape(C2,(3,1))
    P1 = np.dot(K, np.dot(R1, np.concatenate((np.identity(3), -C1), axis=1)))
    P2 = np.dot(K, np.dot(R2, np.concatenate((np.identity(3), -C2), axis=1)))

    p_row1_T = P1[0,:].reshape(1,4)
    p_row2_T = P1[1,:].reshape(1,4)
    p_row3_T = P1[2,:].reshape(1,4)
    # print(p_row1_T)
    # print(p_row2_T)
    
    p_dash_row1_T = P2[0,:].reshape(1,4)
    p_dash_row2_T = P2[1,:].reshape(1,4)
    p_dash_row3_T = P2[2,:].reshape(1,4)
    
    # Cross product
    #          a2b3 − a3b2
    # a X b =  a3b1 − a1b3
    #          a1b2 − a2b1
    pts_3d = []
    for i,_ in enumerate(src_points):
        x = src_points[i,0]
        y = src_points[i,1]

        x_dash = dst_points[i,0]
        y_dash = dst_points[i,1]
        # This equation { [x y z] cross [p1T.X p2T.X p3T.X] = [0 0 0] } has three rows but provides only two constraints on X
        # since each row can be expressed as a linear combination of the other two.
        # One correspondence gives 2 equations (dash -> second point)
        A_ = []

        a2b3_a3b2 = ((y * p_row3_T) - p_row2_T)
        a3b1_a1b3 = (p_row1_T - (x * p_row3_T))
        a2b3_a3b2_dash = ((y_dash * p_dash_row3_T) - p_dash_row2_T)
        a3b1_a1b3_dash = (p_dash_row1_T - (x_dash * p_dash_row3_T))

        A_.append([a2b3_a3b2, a3b1_a1b3, a2b3_a3b2_dash, a3b1_a1b3_dash])

        A = np.array(A_).reshape(4,4)
        # AX = 0
        _, _, vt_ = np.linalg.svd(A)
        vt_ = vt_.T
        x_3d = vt_[:,-1]
        # x_3d = x_3d/x_3d[-1]
        pts_3d.append(x_3d)

    list_3d_points = np.array(pts_3d)
    return list_3d_points

