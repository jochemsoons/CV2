import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time
import numpy_indexed as npi
from helpers import detect_interest_points, match_keypoints, ratio_test
from RANSAC import *
from main import *

def get_image_path(i):

    if i < 10:
        return 'Data/House/frame0000000{}.png'.format((i))
    else:
        return 'Data/House/frame000000{}.png'.format((i))

def connect_first_and_last_frame(pv_matrix):
    # connect points frame 1-2 to points frame 49-1
    print('merge points frame 49 to frame 1 in point view matrix...')
    pv_matrix_copy = pv_matrix
    row_indexes_to_delete = []
    for i in range(len(pv_matrix)):
        if all(v == 0 for v in pv_matrix[i][-2:]) and all(v != 0 for v in pv_matrix[i][:2]):
            first_point = pv_matrix[i][:2]
            for j in range(len(pv_matrix)):
                if all(v != 0 for v in pv_matrix[j][-2:]) and all(v == 0 for v in pv_matrix[j][:2]):
                    last_point = pv_matrix[j][-2:]
                    # if point in frame 1 is same as in frame 49, append point row frame 49 to point row frame 1
                    if list(first_point) == list(last_point):
                        row_indexes_to_delete.append(j)
                        first_index_to_insert = next(x[0] for x in enumerate(pv_matrix[j]) if x[1] > 0)
                        pv_matrix_copy[i][first_index_to_insert:] = pv_matrix_copy[j][first_index_to_insert:]
    
    pv_matrix = np.delete(pv_matrix_copy, row_indexes_to_delete, 0)
    for j in range(len(pv_matrix)):
        if all(v != 0 for v in pv_matrix[j][-2:]):
            pv_matrix[j][0:2] = pv_matrix[j][-2:]
    pv_matrix = pv_matrix[:,:-2]
    return pv_matrix

def plot_pv_matrix(pv_matrix, print_values):

    # translate point view matrix to get correct shape and save the matrix
    print('size of point view matrix:', np.shape(pv_matrix))
    plt.imshow(np.array(pv_matrix, dtype=bool), cmap='hot', interpolation='nearest', aspect='auto')
    plt.xlabel('points')
    plt.ylabel('views (2 rows are x and y coordinates of 1 view)')
    plt.title('point view matrix for house images')
    plt.savefig('point_view_matrix_coor_{}_des_{}_des_and_coor_{}.png'.format(print_values[0], print_values[1], print_values[2]))
    np.savetxt('point_view_matrix_coor_{}_des_{}_des_and_coor_{}.txt'.format(print_values[0], print_values[1], print_values[2]), pv_matrix, fmt='%.2f')
    return pv_matrix

def point_view_matrix(sift, matcher, num_matches, num_iter, subset_size, max_ransac_distance, max_des_distance,
                        match_coor_only, match_des_only, match_des_and_coor):
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
    print_values = [match_coor_only, match_des_only, match_des_and_coor]

    if num_matches > 500:
        print('num_matches argument limited to 500. If you want all matches added per frame pair, set num_matches=-1.')
        return

    index = 2
    # Loop over consequetive house frame pairs
    for i in range(1, 50):
        
        # Compute interest points.
        if i == 1:
            kp1, des1 = detect_interest_points(sift, get_image_path(i))
            kp0, des0 = kp1, des1
            kp2, des2 = detect_interest_points(sift, get_image_path(i+1))
        elif i == 49:
            kp1, des1 = kp2, des2
            kp2, des2 = kp0, des0
        else:
            kp1, des1 = kp2, des2
            kp2, des2 = detect_interest_points(sift, get_image_path(i+1))

        # Compute matches across house frame pair
        matches_descriptors, matches_matrix = match_keypoints(matcher, des1, des2, kp1, kp2, num_matches)

        # Determine best matches using RANSAC algorithm
        temp = np.einsum('kli->lki', matches_matrix)
        pts1, pts2 = e2h(matches_matrix[:, 0, :]).T, e2h(matches_matrix[:, 1, :]).T # (3 x num_matches)
        _, _, _, indices = ransac(pts1, pts2, num_iter, subset_size, max_ransac_distance)
        matches_descriptors = [matches_descriptors[idx] for idx in indices]
        matches_matrix = [matches_matrix[idx] for idx in indices]

        if i == 49:
            print('adding {} matches frame pair'.format(len(matches_matrix)), i, "-", 1, "to point view matrix...")
        else:
            print('adding {} matches frame pair'.format(len(matches_matrix)), i, "-", i+1, "to point view matrix...")

        # Initialize the point view matrix and add matches between frame 1-2 to it
        if i == 1:
            shape = np.shape(matches_matrix)
            matrix = np.round(matches_matrix, 2).reshape(shape[0], shape[1]*shape[2])
            shape = np.shape(matrix)
            pv_matrix = np.zeros((shape[0], 2*50))
            pv_matrix[:shape[0], :shape[1]] = matrix
            descriptors = np.array([des1[m.queryIdx] for m in matches_descriptors])

        # Add matching points for consequetive frame pairs 2-3, 3-4 ... 49-1
        else:
            # Case where matching is done based on coordinates
            if match_des_only == False:
                descriptors_img_1 = np.array([des1[m.queryIdx] for m in matches_descriptors])
                for j in range(len(matches_matrix)):
                    match = matches_matrix[j]
                    g = np.where(~(pv_matrix[:, index:index+2] - match[0]).any(axis=1))[0]
                    points_found = len(g)
                    if points_found == 0:
                        match_found = False
                    else:
                        match_found = True
                        if points_found > 1:
                            pv_matrix[random.choice(g)][index+2:index+4] = match[1]
                        else:
                            pv_matrix[g[0]][index+2:index+4] = match[1]

                    if not match_found:
                        # Case: If match based on coordinate is not found, check if any point column has 
                        # similar descriptor. If yes, add point to dedicated column. If not, add new column
                        # For that point to point view matrix. 
                        if match_des_and_coor == True:
                            m = bf.match(descriptors, np.array([descriptors_img_1[j]]))[0]
                            if m.distance < max_des_distance:
                                pv_matrix[m.queryIdx][index+2:index+4] = match[1]
                                descriptors[m.queryIdx] = descriptors_img_1[j]
                            else:
                                new_point_list = np.zeros(2*50)
                                new_point_list[index:index+4] = match.flatten()
                                pv_matrix = np.vstack((pv_matrix, new_point_list))
                                descriptors = np.append(descriptors, [descriptors_img_1[j]], axis=0)
                        
                        # Case: If match based on coordinate is not found, add new column for that point
                        # directly to point view matrix.
                        elif match_coor_only == True:
                            new_point_list = np.zeros(2*50)
                            new_point_list[index:index+4] = match.flatten()
                            pv_matrix = np.vstack((pv_matrix, new_point_list))
            
            # Case where matching is done based on descriptors only.
            elif match_des_only == True:
                descriptors_img_1 = np.array([des1[m.queryIdx] for m in matches_descriptors])
                for j in range(len(matches_matrix)):
                    match = matches_matrix[j]
                    m = bf.match(descriptors, np.array([descriptors_img_1[j]]))[0]
                    if m.distance < max_des_distance:
                        pv_matrix[m.queryIdx][index+2:index+4] = match[1]
                        descriptors[m.queryIdx] = descriptors_img_1[j]
                    else:
                        new_point_list = np.zeros(2*50)
                        new_point_list[index:index+4] = match.flatten()
                        pv_matrix = np.vstack((pv_matrix, new_point_list))
                        descriptors = np.append(descriptors, [descriptors_img_1[j]], axis=0)

            index += 2

    # connect points in last and first row pv matrix
    pv_matrix = connect_first_and_last_frame(pv_matrix)

    # translate point view matrix to get correct shape and save the matrix
    pv_matrix = np.transpose(pv_matrix)
    
    # plot and save pv matrix
    plot_pv_matrix(pv_matrix, print_values)
    
    return pv_matrix
