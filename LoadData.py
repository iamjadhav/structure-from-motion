import glob
import cv2
import numpy as np


def load_dataset(folder, number_of_images):
    '''
    Loading the stereo images
    '''
    image_list = []
    for i in range(number_of_images):
        for file in glob.glob(folder+str(i+1)+'.jpg'):
            image = cv2.imread(file)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows
            if image is not None:
                image_list.append(image)
            else:
                print('This image cannot be read.')
    return image_list


def load_data(folder, number_of_images):
    '''
    Loading the point correspondences from the matching.txts
    '''
    descriptors = []
    x_coors = []
    y_coors = []
    coor_flags = []

    for i in range(1, 2):
        try:
            file = open(folder + '/matching' + str(i) + '.txt', 'r')
        except FileNotFoundError:
            print('File cannot be read !')

        for row_num, each_row in enumerate(file):
            if row_num == 0:
                print(int(each_row.split(':')[1]))
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
                descriptors.append([red_val,green_val,blue_val])
                # Current image feature coordinates
                source_x = feature_matches[4]
                source_y = feature_matches[5]
                # Copy them to the corresponding number in the zero arrays
                x_arr[0][i-1] = source_x
                y_arr[0][i-1] = source_y
                flags_arr[0][i-1] = 1
                # print(x_arr)
                
                next_im = 1
                for _ in range(int(n_matches) - 1):
                    # Next match image id and X, Y coors 
                    # [n_features R G B (u_current image) (v_current image) (image id) (u_image id image) (v_image id image) (image id) (u_image id image) (v_image id image)]
                    image_id = int(feature_matches[5+next_im])
                    x_next_im = feature_matches[6+next_im]
                    y_next_im = feature_matches[7+next_im]
                    next_im += 3
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
    print(x_coors)

    return x_coors, y_coors, coor_flags, descriptors