import numpy as np
from RANSAC import ransac
from helpers import compute_matrix_A


def normalize_F(pts1, pts2):
    mean_x = np.mean(pts1[0])
    mean_x_ = np.mean(pts2[0])
    mean_y = np.mean(pts1[1])
    mean_y_ = np.mean(pts2[1])

    d = np.mean(np.sqrt((pts1[0] - mean_x)**2 + (pts1[1] - mean_y)**2))
    d_ = np.mean(np.sqrt((pts2[0] - mean_x_)**2 + (pts2[1] - mean_y_)**2))

    T = np.array([[np.sqrt(2)/d, 0, -mean_x * np.sqrt(2)/d],
                [0, np.sqrt(2)/d, -mean_y * np.sqrt(2)/d],
                [0, 0, 1]])
    T_ = np.array([[np.sqrt(2)/d_, 0, -mean_x_ * np.sqrt(2)/d_],
                [0, np.sqrt(2)/d_, -mean_y_ * np.sqrt(2)/d_],
                [0, 0, 1]])
    
    return T_, T, T @ pts1, T_ @ pts2

def eight_point_algorithm(pts1, pts2, normalize, use_ransac, num_iter, subset_size, max_distance):

    if len(pts1[0]) < 8:
        print("not enough matches. Need at least 8.")
        return 0
    if normalize:
        T_, T, pts1, pts2 = normalize_F(pts1, pts2)
    if use_ransac:
        F, pts1, pts2, _ = ransac(pts1, pts2, num_iter, subset_size, max_distance)

    A = compute_matrix_A(pts1, pts2)
    _, _, Vt = np.linalg.svd(A) # Decompose A
    F = np.reshape(Vt[-1], (3, 3)) # Reshape entries of V to 3x3 to get F

    # Set rank F to 2
    U_f, D_f, Vt_f = np.linalg.svd(F)
    D_f[2] = 0
    F = U_f @ (np.diag(D_f) @ Vt_f)
    
    if normalize:
        F = T_.T @ (F @ T)

    return F
