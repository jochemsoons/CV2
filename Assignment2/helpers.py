import numpy as np
import cv2

def sampson_distance(pts1, pts2, F):
    Fpts1, Fpts2 = F @ pts1, F @ pts2
    denom = Fpts1[0]**2 + Fpts1[1]**2 + Fpts2[0]**2 + Fpts2[1]**2
    return np.diag(pts1.T @ (F @ pts2))**2 / denom

def compute_matrix_A(pts1, pts2):
    n = len(pts1[0])
    A = np.ones((n, 9)) # Initiate A

    for i in range(n):
        x, x_, y, y_  = pts1[0][i], pts2[0][i], pts1[1][i], pts2[1][i]
        A[i] = [x*x_, x*y_, x, y*x_, y*y_, y, x_, y_, 1]
    return A

def detect_interest_points(sift, path_to_img):
    img = cv2.imread(path_to_img, 0) # Load in image

    # Find and return SIFT keypoints and compute their descriptors    
    return sift.detectAndCompute(img, None)

# Function to match keypoints of two images.
def match_keypoints(matcher, des1, des2, kp1, kp2, num_matches):
    matches = matcher.knnMatch(des1, des2, k=2) # Match descriptors
    
    good_matches = []
    threshold = 0.1
    i = 0
    # check if all or subselection of points is wanted
    if num_matches == -1:
        num_matches = len(matches)

    # Continue applying the ratio test with increasing threshold until enough matches found.
    while len(good_matches) < num_matches:
        good_matches = ratio_test(matches, threshold)
        threshold *= 1.01
        i += 1
        # Stop after 50 tries (not enough matches, even with high threshold.)
        if i >= 300:
            print("No good matches found. Aborting program")
            return None 

    sorted_matches = sorted(good_matches, key = lambda x: x.distance) # Sort by distance
    return sorted_matches, np.array([((kp1[m.queryIdx].pt), (kp2[m.trainIdx].pt)) for m in sorted_matches[:]])

# Apply Lowe's ratio test to filter out good matches.
def ratio_test(matches, threshold):
    good_matches = []
    for m_1, m_2 in matches:
        if m_1.distance < threshold * m_2.distance:
            good_matches.append(m_1)
    return good_matches