import numpy as np


def estimate_fundamental_matrix(source_8_points, dest_8_points):
    '''
    To estimate Fundamental Matrix using the 8-point algorithm
    '''
    x_1, x_2 = np.array(source_8_points), np.array(dest_8_points)
    # print(x_1, x_2)
    #calculating the fundamental matrix by shifting the origin to mean, then applying svd and unnormalising the points to get F
    # Normalized 8-point algorithm
    # https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm
    # https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
    if len(x_1) > 7:
        ###### Compute centroid of all corresponding points #####
        # Mean of all points (src&dst)
        src_mean = np.mean(x_1, axis=0)
        dst_mean = np.mean(x_2, axis=0)
        # print(src_mean, dst_mean)
        ##### Recenter: OG - mean points #####
        # Unpacking X, Ys for src and dest points
        src_mean_x, src_mean_y = src_mean[0], src_mean[1]
        dst_mean_x, dst_mean_y = dst_mean[0], dst_mean[1]
        # print(src_mean_x, src_mean_y)
        
        # Recentering <-> transforming points
        src_tilda_x, src_tilda_y = x_1[:,0] - src_mean_x, x_1[:,1] - src_mean_y
        dst_tilda_x, dst_tilda_y = x_2[:,0] - dst_mean_x, x_2[:,1] - dst_mean_y
        # print(src_tilda_x, src_tilda_y)
        # Scale factors
        s_src = (2 / (np.mean(src_tilda_x**2 + src_tilda_y**2))) ** 0.5
        s_dst = ( 2 / (np.mean(dst_tilda_x**2 + dst_tilda_y**2))) ** 0.5
        # print(s_src, s_dst)
       
        ##### Scale matrices and Translation matrices (diagonal) with mean X, Ys
        scale_src = np.array([[s_src, 0, 0],[0 , s_src, 0],[0, 0, 1]])
        scale_dst = np.array([[s_dst, 0, 0],[0 , s_dst, 0],[0, 0, 1]])

        trans_src = np.array([[1, 0, -src_mean_x],[0, 1, -src_mean_y],[0, 0, 1]])
        trans_dst = np.array([[1, 0, -dst_mean_x],[0, 1, -dst_mean_y],[0, 0, 1]])
        # print(trans_src, trans_dst)
        
        ##### T_a (source) and T_b (dest) matrices
        T_a = scale_src.dot(trans_src)
        T_b  = scale_dst.dot(trans_dst)
        # print(T_a, T_b)

        ##### X Normalized - src, dst #####
        # X_norm = T_a/b * x
        x_1_norm = np.column_stack((x_1, np.ones(len(x_1))))
        x_1_norm = T_a@x_1_norm.T
        x_1_norm = x_1_norm.T
        x_2_norm = np.column_stack((x_2, np.ones(len(x_2))))
        x_2_norm = T_b@x_2_norm.T
        x_2_norm = x_2_norm.T
        # print(x_1_norm, x_2_norm)
       
        # A (8X9) matrix
        Alist = []
        for i in range(x_1_norm.shape[0]):
            X1, Y1 = x_1_norm[i][0], x_1_norm[i][1]
            X2, Y2 = x_2_norm[i][0], x_2_norm[i][1]
            Alist.append([X2*X1 , X2*Y1 , X2 , Y2 * X1 , Y2 * Y1 ,  Y2 ,  X1 ,  Y1, 1])
        A = np.array(Alist)
        # print(A)
     
        U, sigma, VT = np.linalg.svd(A)  
        v = VT.T
        f_val = v[:,-1]
        f_mat = f_val.reshape((3,3))
        # print(f_mat)

        Uf, sigma_f, Vf = np.linalg.svd(f_mat)
        ## Enforcing the rank 2 constraint
        sigma_final = np.diag(sigma_f)
        sigma_final[2,2] = 0
        # un-normalizing
        f_main = np.dot(Uf , np.dot(sigma_final, Vf))
        # print('F main: ', f_main)

        f_unnorm = np.dot(T_b.T, np.dot(f_main, T_a))
        f_unnorm = f_unnorm/f_unnorm[-1,-1]
        # print('Unnormalized F', f_unnorm)
    else:
        return None
  
    return f_unnorm
