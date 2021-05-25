import numpy as np
import random
from helpers import sampson_distance, compute_matrix_A

def ransac(pts1, pts2, num_iter, subset_size, max_distance):
    max_inliers = 0 # Initiate maximum found inliers
    for _ in range(num_iter):
    
        subset_indices = random.sample(range(0, len(pts1[0])), subset_size)
        pts1_subset, pts2_subset = pts1.T[subset_indices].T, pts2.T[subset_indices].T
        A = compute_matrix_A(pts1_subset, pts2_subset)
        _, _, Vt = np.linalg.svd(A) # Decompose A
        F = np.reshape(Vt[-1], (3, 3)) # Reshape entries of V to 3x3 to get F

        # Set rank F to 2
        U_f, D_f, Vt_f = np.linalg.svd(F)
        D_f[2] = 0
        F = U_f @ (np.diag(D_f) @ Vt_f)

        # calculate inliers given calculated model
        distances = sampson_distance(pts1, pts2, F)
        inlier_indices = np.where(distances <= max_distance)[0]
        num_inliers = len(inlier_indices)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = F
            max_inlier_indices = inlier_indices
    
    if max_inliers == 0:
        return 0, 0, 0
    return best_F, pts1.T[max_inlier_indices].T, pts2.T[max_inlier_indices].T, max_inlier_indices
