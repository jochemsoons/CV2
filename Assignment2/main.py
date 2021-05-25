import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from point_view_matrix import *
from fundamental_matrix import eight_point_algorithm
from structure_from_motion import structure_from_motion
from helpers import *
import time

# Convert euclidean coordinates to homogeneous
def e2h(data):
    homogeneous_data = np.ones((data.shape[0], 3))
    homogeneous_data[:,0:2] = data
    return homogeneous_data

# Convert homogeneous coordinates to euclidean
def h2e(data):
    last_col = data[:, [-1]]
    data = data / last_col
    return data[:, 0:-1]

# This code is credit to opencv-python-tutroals
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, thickness=1)
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)
    return img1, img2

def produce_results_3(normalize, use_ransac, num_matches, num_iter, subset_size, max_distance, additional_plots):
    sift = cv2.SIFT_create() # Initiate SIFT detector
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) # Initiate matching model

    # Get images
    img1 = cv2.imread('Data/House/frame00000001.png', 0)
    img2 = cv2.imread('Data/House/frame00000002.png', 0)

    # Compute keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Compute matches
    _, matches = match_keypoints(matcher, des1, des2, kp1, kp2, num_matches)

    # Split matches into two sepearte lists and convert to homogeneus coordinates
    pts1, pts2 = e2h(matches[:, 0, :]).T, e2h(matches[:, 1, :]).T # (3 x num_matches)

    if additional_plots:
        matches = matcher.knnMatch(des1, des2, k=2) # Match descriptors
        all_matches = []
        for m_1, _ in matches:
            all_matches.append(m_1)
        sorted_all_matches = sorted(all_matches, key = lambda x: x.distance) # Sort by distance
        best_100_matches = np.array([((kp1[m.queryIdx].pt), (kp2[m.trainIdx].pt)) for m in sorted_all_matches[:100]])
        or_pts1, or_pts2 = e2h(best_100_matches[:, 0, :]).T, e2h(best_100_matches[:, 1, :]).T

    # Compute fundamental matrix
    tic = time.perf_counter()
    F = eight_point_algorithm(pts1, pts2, normalize, use_ransac, num_iter, subset_size, max_distance)
    toc = time.perf_counter()

    if additional_plots:
        error = []
        for i in range(100):
            error.append(np.abs(or_pts2[:, i] @ (F @ or_pts1[:, i].T)))
        return np.mean(error), toc - tic

    # Convert points back to eaclidain coordinates
    pts1, pts2 = h2e(pts1.T)[:, :], h2e(pts2.T)[:, :] # (num_matches x 2)

    # Plot lines
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    img_to_show_1, _ = drawlines(img1, img2, lines1.reshape(-1, 3), pts1, pts2)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img_to_show_2, _ = drawlines(img2, img1, lines2.reshape(-1, 3), pts2, pts1)
    mpl.use('tkagg')
    plt.subplot(121), plt.imshow(img_to_show_1)
    plt.subplot(122), plt.imshow(img_to_show_2)
    # plt.show()
    plt.savefig("Epilines_normal{}_ransac{}_numMatches{}.jpg".format(normalize, use_ransac, num_matches))

def create_plot_of_error_and_time_against_nummatches():

    sift = cv2.SIFT_create() # Initiate SIFT detector
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) # Initiate matching model

    # Set parameters
    num_iter, subset_size, max_distance, additional_plots = 50, 20, 1e-8, True
    x, y1, y2, y3, time_1, time_2, time_3 = [], [], [], [], [], [], []
    for i in range(8, 540, 1):
        print("{}/540".format(i))
        x.append(i)

        # Set parameters for unnormalized without ransac
        normalize, use_ransac = False, False
        temp, time_temp = produce_results_3(normalize, use_ransac, i, num_iter, subset_size, max_distance, additional_plots)
        y1.append(temp)
        time_1.append(time_temp)

        # Set parameters for normalized without ransac
        normalize, use_ransac = True, False
        temp, time_temp = produce_results_3(normalize, use_ransac, i, num_iter, subset_size, max_distance, additional_plots)
        y2.append(temp)
        time_2.append(time_temp)

        normalize, use_ransac = True, True
        temp, time_temp = produce_results_3(normalize, use_ransac, i, num_iter, subset_size, max_distance, additional_plots)
        y3.append(temp)
        time_3.append(time_temp)

    print("Minimum error for unnormalized without RANSAC model is {} for num_matches is {}".format(min(y1), 8+int(np.argwhere(y1 == min(y1)))))
    print("Minimum error for normalized without RANSAC model is {} for num_matches is {}".format(min(y2), 8+int(np.argwhere(y2 == min(y2)))))
    print("Minimum error for normalized with RANSAC model is {} for num_matches is {}".format(min(y3), 8+int(np.argwhere(y3 == min(y3)))))

    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    ax1.plot(x, y1, c='r', label="Unnormalized without RANSAC", alpha=1, linewidth=0.5)
    ax1.plot(x, y2, c='g', label="Normalized without RANSAC", alpha=1, linewidth=0.5)
    ax1.plot(x, y3, c='b', label="Normalized with RANSAC", alpha=1, linewidth=0.5)
    plt.xlabel("Number of samples")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    ax2.plot(x, time_1, c='r', label="Unnormalized without RANSAC", alpha=1, linewidth=0.5)
    ax2.plot(x, time_2, c='g', label="Normalized without RANSAC", alpha=1, linewidth=0.5)
    ax2.plot(x, time_3, c='b', label="Normalized with RANSAC", alpha=1, linewidth=0.5)
    plt.xlabel("Number of samples")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    question = str(input("Enter the question number for which results need to be produced. \nQuestions can be 3.4, 4.2, 5.2 or 6.\nIf you want to reproduce additional plots in the report (sec. 3), type plots.\nInpute here: "))
    while question not in ['3.4', '4.2', '5.2', '6', 'plots']:
        question = str(input("Not a valid question number. Needs to be 3.4, 4.2, 5.2, 6 or plots: "))

    sift = cv2.SIFT_create() # Initiate SIFT detector
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) # Initiate matching model

    if question == '3.4':
        # Set parameters
        num_iter, subset_size, max_distance, additional_plots = 50, 8, 1e-8, False

        # Set parameters for unnormalized without ransac
        normalize, use_ransac, num_matches = False, False, 338
        produce_results_3(normalize, use_ransac, num_matches, num_iter, subset_size, max_distance, additional_plots)

        # Set parameters for normalized without ransac
        normalize, use_ransac, num_matches = True, False, 97
        produce_results_3(normalize, use_ransac, num_matches, num_iter, subset_size, max_distance, additional_plots)

        normalize, use_ransac, num_matches = True, True, 80
        produce_results_3(normalize, use_ransac, num_matches, num_iter, subset_size, max_distance, additional_plots)

    elif question == '4.2':
        tic = time.perf_counter()

        num_matches = -1 # number of matches between 2 frames we want to be returned. set to -1 for all matches.
        num_iter = 100 # number of iterations for RANSAC to find inlier matches between frames.
        subset_size = 8 # number of points used to create transformation matrix for RANSAC to find inliers.
        max_ransac_distance = 0.001 # maximum distance between frame points in RANSAC to be considered as inlier.
        max_des_distance = 150 # maximum distance between two descriptors to be considered the same. Applicable if we match based on descriptors.

        # Only set 1 out of 3 on True
        match_coor_only = False # Set to True if you want to create point view matrix based on matching coordinates across frames only.
        match_des_only = True # Set to True if you want to create point view matrix based on matching descriptors across frames only.
        match_des_and_coor = False # Set to True if you want to create point view matrix based on matching coordinates and descriptors across frames.

        point_view_matrix(sift, matcher, num_matches, num_iter, subset_size, max_ransac_distance, max_des_distance, match_coor_only, 
        match_des_only, match_des_and_coor)

        toc = time.perf_counter()
        print(f"Downloaded the tutorial in {toc - tic} seconds")

    elif question == '5.2':

        num_matches = 200 # number of matches between 2 frames we want to be returned. set to -1 for all matches.
        num_iter = 100 # number of iterations for RANSAC to find inlier matches between frames.
        subset_size = 8 # number of points used to create transformation matrix for RANSAC to find inliers.
        max_ransac_distance = 0.01 # maximum distance between frame points in RANSAC to be considered as inlier.
        max_des_distance = 250 # maximum distance between two descriptors to be considered the same. Applicable if we match based on descriptors.

        # Only set 1 out of 3 on True
        match_coor_only = False # Set to True if you want to create point view matrix based on matching coordinates across frames only.
        match_des_only = False # Set to True if you want to create point view matrix based on matching descriptors across frames only.
        match_des_and_coor = True # Set to True if you want to create point view matrix based on matching coordinates and descriptors across frames.

        # Perform SFM on provided dense PVM.
        pvm = np.loadtxt('PointViewMatrix.txt')
        structure_from_motion(pvm, method='dense_matrix')
        # Perform SFM on the dense sub block techniques
        for method in ['single_block', 'block_steps']:
            for m in [3, 4]:
                if method == 'single_block':
                    num_matches = 300
                else:
                    num_matches = 200
                
                pvm = point_view_matrix(sift, matcher, num_matches, num_iter, 
                                        subset_size, max_ransac_distance, 
                                        max_des_distance, match_coor_only, 
                                        match_des_only, match_des_and_coor)

                structure_from_motion(pvm, method, m)
    elif question == '6':
        num_matches = 200 # number of matches between 2 frames we want to be returned. set to -1 for all matches.
        num_iter = 100 # number of iterations for RANSAC to find inlier matches between frames.
        subset_size = 8 # number of points used to create transformation matrix for RANSAC to find inliers.
        max_ransac_distance = 0.1 # maximum distance between frame points in RANSAC to be considered as inlier.
        max_des_distance = 250 # maximum distance between two descriptors to be considered the same. Applicable if we match based on descriptors.

        # Only set 1 out of 3 on True
        match_coor_only = False # Set to True if you want to create point view matrix based on matching coordinates across frames only.
        match_des_only = False # Set to True if you want to create point view matrix based on matching descriptors across frames only.
        match_des_and_coor = True # Set to True if you want to create point view matrix based on matching coordinates and descriptors across frames.

        # Additional improvement: removing affine ambiguity.
        pvm = np.loadtxt('PointViewMatrix.txt')
        structure_from_motion(pvm, method='dense_matrix', eliminate_affine=True)
        for method in  ['single_block', 'block_steps']:
            if method == 'single_block':
                num_matches = 300
            else:
                num_matches = 200
            
            pvm = point_view_matrix(sift, matcher, num_matches, num_iter, 
                                    subset_size, max_ransac_distance, 
                                    max_des_distance, match_coor_only, 
                                    match_des_only, match_des_and_coor)
            structure_from_motion(pvm, method, m=4, eliminate_affine=True)
    elif question == 'plots':
        create_plot_of_error_and_time_against_nummatches()