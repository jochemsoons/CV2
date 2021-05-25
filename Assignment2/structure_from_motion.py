import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from helpers import *
from ICP import icp
from scipy.linalg import orthogonal_procrustes


def visualize_structure(S, title, method):    
    print("Visualizing 3D structure of shape: {}".format(np.shape(S)))
    # Remove outiers from stitched blocks to clarify results.
    if method == 'block_steps' or method == 'single_block':
        S_new = []
        for point in S:
            if abs(point[2]) < 1:
                S_new.append(point)
        S_new = np.asarray(S_new)
    else:
        S_new = S
    # Extract X, Y and Z vectors from S.
    X, Y, Z = S_new[:,0], S_new[:,1], S_new[:,2]
    # Create and show a 3D plot using matplotlib.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(title)
    ax.scatter3D(X, Y, Z)
    plt.show()
    
# Function to extract a single dense block given a selection of rows within the PVM.
def extract_dense_block(point_view_rows):    
    non_zero_indices = []
    non_zero_cols = []
    # Loop over the columns of the PVM.
    for j, col in enumerate(point_view_rows.T):
        # If no values in the column are zero...
        if np.all(col):
            # Save the column and the column index.
            non_zero_cols.append(col)
            non_zero_indices.append(j)
    non_zero_cols = np.asarray(non_zero_cols).T
    return non_zero_cols, non_zero_indices

# Function to compute the M and S matrix given a dense sub block.
def compute_motion_structure(dense_block, eliminate_affine_ambiguity):
    # Normalize the dense block.
    mean = np.mean(dense_block, axis=1, keepdims=True)
    D_normalized = dense_block - mean

    # Perform SVD to obtain U, W and V.T
    U, W, Vt = np.linalg.svd(D_normalized, full_matrices=True)

    # Ensure the measurement matrix has rank 3.
    U_3 = U[:, :3]
    W_3 = np.diag(W[:3])
    V_3 = Vt[:3, :]

    # Compute M and S.
    M = U_3 @ np.sqrt(W_3)
    S = np.sqrt(W_3) @ V_3

    # If we want to eliminate the affine ambiguity.
    if eliminate_affine_ambiguity:
        # Calculate the pseudo-inverse M^(-1) and M.T^(-1)
        M_inv = np.linalg.pinv(M)
        M_inv_T = np.linalg.pinv(M.T)

        # L is the product of M^(-1) and M.T^(-1)
        L = M_inv @ M_inv_T

        # Recover C from L through Cholesky decomposition.
        C = np.linalg.cholesky(L)

        # Update M and S using C.
        M = M @ C
        S = np.linalg.inv(C) @ S
    return M, S

def procrustes(S_prev, S_new):
    # translate all the data to the origin
    S_prev -= np.mean(S_prev, 0)
    S_new -= np.mean(S_new, 0)

    norm1 = np.linalg.norm(S_prev)
    norm2 = np.linalg.norm(S_new)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(S_prev*S_new') = 1
    S_prev /= norm1
    S_new /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(S_prev, S_new)
    return R, s


def structure_from_motion(point_view_matrix, method='block_steps', m=4, use_icp=False, eliminate_affine=False):
    if method not in ['single_block', 'block_steps', 'dense_matrix']:
        raise AssertionError("No valid method. Use single_block, block_steps or dense_matrix as method.")

    print("Eliminate affine ambiguity: {}".format(eliminate_affine))
    print("Size of m to extract blocks: {}".format(m))
    pvm_rows, _ = np.shape(point_view_matrix)

    # Method 1: performing SFM using one single dense block.
    if method == 'single_block':
        print("Extracting the best single dense block...")
        most_points = 0
        dense_block = None
        i = 0
        while i + 2*m <= pvm_rows:
            # Extract a 2m * n (sparse) block from the PVM.
            point_view_rows = point_view_matrix[i: i + 2*m, :]
            # Extract only the dense columns from the block.
            dense_block, dense_col_indices = extract_dense_block(point_view_rows)
            if len(dense_col_indices) >= 3 and len(dense_col_indices) > most_points:
                best_block = dense_block
                most_points = len(dense_col_indices)
            i += 2
        if dense_block is not None:
            M, S = compute_motion_structure(dense_block, eliminate_affine_ambiguity=eliminate_affine)
            visualize_structure(S.T, 'Structure from single dense block', method)
        else:
            print("No dense blocks with at least 3 points found. Aborting.")
    
    # Method 2: performing SFM iteratively on dense blocks and stitch results together.
    elif method == 'block_steps':
        print("Taking steps in extracting dense blocks...")
        i = 0
        first_iter = True

        # Continue until the final row of the PVM.
        while i + 2*m <= pvm_rows:
            # Extract a 2m * n (sparse) block from the PVM.
            point_view_rows = point_view_matrix[i: i + 2*m, :]

            # Extract only the dense columns from the block.
            dense_block, dense_col_indices = extract_dense_block(point_view_rows)

            # Only continue with factorization when there are at least 3 unique points in the block.
            if len(dense_col_indices) >= 3:
                # Compute M and S.
                M, S = compute_motion_structure(dense_block, eliminate_affine_ambiguity=eliminate_affine)
                S = S.T
                # If we are in our first iteration...
                if first_iter:
                    # Save the current variables for next iteration.
                    main_view = S
                    S_prev = S
                    prev_cols = dense_col_indices
                    first_iter = False
                
                # If we are not in the first iteration...
                else:
                    # Extract the common columns (points) from both point clouds.
                    common_cols = sorted(list(set(prev_cols) & set(dense_col_indices)))
                    
                    # Extract the corresponding indices for the previous and new S matrix.
                    indices_prev = [i for i, item in enumerate(prev_cols) if item in common_cols]
                    indices_current = [i for i, item in enumerate(dense_col_indices) if item in common_cols]
                    # Also determine which points are new (not seen before).
                    indices_current_new = [i for i, item in enumerate(dense_col_indices) if item not in common_cols]
                    
                    # Take the corresponding subselections of S.
                    S_prev_intersect = S_prev[indices_prev, :]
                    S_intersect = S[indices_current, :]
                    S_new = S[indices_current_new, :]
                    
                    # Ensure we have more than one shared point between blocks.
                    if len(S_prev_intersect) > 1:
                        # If we want to use ICP for alginment...
                        if use_icp:
                            # Compute the transformation matrix R and translation vector t between the prev and new S with Iterative Closest Point.
                            R, t, _, _ = icp(S_intersect, S_prev_intersect, epsilon=1e-5, sample_size=len(S_prev_intersect)-1, closest_point_search='KDTree', sampling_method='c', noise_sigma=None, sample_target=False, max_distance=None)
                            S_new_transformed = np.dot(S_new, R.T) + t
                        # Otherwise compute the transformation matrix R between the prev and new S through Procrustes analysis.
                        else:
                            R, s = procrustes(S_prev_intersect, S_intersect)
                            # Transform the new 3D structure points to align with the main view and add them to it.
                            S_new_transformed = np.dot(S_new, R.T) * s
                        main_view = np.concatenate((main_view, S_new_transformed), axis=0)
                        
                    # Set the variables for the next iteration.
                    S_prev = S
                    prev_cols = dense_col_indices
                i += 2
                continue
            i += 2
        # Visualize the 3D structure of the stitched S matrices.
        try:
            visualize_structure(main_view, 'Structure from stitched dense blocks', method)
        except:
            print("No dense blocks found containing at least 3 points. Aborting.")

    # Method 3: performing SFM on the provided dense Point View Matrix.
    elif method == 'dense_matrix':
        print("Performing SFM on dense Point View Matrix...")
        M, S = compute_motion_structure(point_view_matrix, eliminate_affine_ambiguity=eliminate_affine)
        visualize_structure(S.T, 'Structure from dense PVM', method)
        