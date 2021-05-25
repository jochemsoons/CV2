import os
import numpy as np
import open3d as o3d
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import sklearn.neighbors
import scipy.io
import csv
from sklearn.cluster import KMeans, MiniBatchKMeans
import time

############################
#   Load Data              #
############################

# Function to load in and preprocess point clouds
def load_data(path, normal=True):

    if normal == True:
        pcd = o3d.io.read_point_cloud(path)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)

    data, counter = [], 0
    with open(path) as f:
        for line in f:
            if counter < 11: # Ignore first 10 lines
                counter += 1
            else:
                for floats in line.split(','):
                    
                    # Ignore 'nan' values
                    if 'nan' not in floats:

                        # Get point into format of list of floats
                        point = [float(x) for x in floats.split()][0:3]

                        # Only add points that are closer than two meters away
                        if point[2] < 1.5 and point[2] > 0.59:
                            data.append(point)
                        else:
                            normals = np.delete(normals, counter, axis=0)
                    else:
                        normals = np.delete(normals, counter, axis=0)

    data = np.array(data) # Convert to numpy array
    if normal == True:
        return data, normals
    return data

############################
#     ICP                  #
############################

# Function to find closest point in set of points 'B' to 'point'
def brute_force_closest_point(A, B, sampling_method):
    
    if sampling_method == "a":
        distances_list, points = [], []
        for i, point in enumerate(A):
            # Compute distances from all points in 'B' to 'point'
            distances = np.sum((point-B)**2, axis=1)
            index = np.where(distances == np.amin(distances))
            distances_closest_points = distances[index]

            # Return point in 'B' corresponding to lowest distance
            if len(index) >= 2:
                temp_index = random.randint(0, len(index)-1)
                points.append(index[temp_index][0])
                distances_list.append(distances_closest_points[temp_index][0])
            else:
                points.append(index[0][0])
                distances_list.append(distances_closest_points[0])

        return np.array(distances_list), np.array(points)
    else:
        distances = np.sum((A[:,None] - B)**2, axis=2)
        return np.amin(distances, axis=1), np.argmin(distances, axis=1)

# Return Root Mean Square
def rms(A1, A2):
    return np.sqrt(((A1 - A2) ** 2).mean())

def get_transform(A1, A2, weights=[]):
    # Compute centered matrices
    A1_centroids, A2_centroids = np.mean(A1, axis=0), np.mean(A2, axis=0)
    A1_centered, A2_centered = A1 - A1_centroids, A2 - A2_centroids

    # Compute rotation matrix
    if len(weights) == 0:
        S = A1_centered.T @ A2_centered
    else:
        S = A1_centered.T @ np.identity(len(A1_centered))*weights @ A2_centered
    U, _, Vt = np.linalg.svd(S)

    diag = np.identity(len(Vt[0]))
    diag[-1][-1] = np.linalg.det(Vt.T@U.T)
    R = Vt.T @ diag @ U.T

    # Compute translation t
    t = A2_centroids - (R@A1_centroids)

    return R, t

# Show point clouds
def show_point_cloud(data, method):
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(data)
    ## Uncomment to write to .ply file.
    # o3d.io.write_point_cloud(method + "mat.ply", vis_pcd)
    o3d.visualization.draw_geometries([vis_pcd])

# Used to load in xyz files.
def read_xyz(path_to_pc):
    x = []
    with open(path_to_pc) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            x.append(list(map(float, row)))
    return x

# Use this function to create simple test point clouds.
def create_test_points_box():
    points = []
    for x in range(20, 80, 3):
        for y in range(20, 70, 3):
            for z in range(20, 50, 3):
                points.append([x, y, z])
    for x in range(80, 120, 3):
        for y in range(20, 40, 3):
            for z in range(20, 50, 3):
                points.append([x, y, z])
    A1 = np.array(points)

    theta = np.radians(10)
    rotation_matrix = [[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]]
    A2 = np.dot(A1, rotation_matrix)

    return A1, A2

# Use to create test_points based on our own donwloaded/created/loaded data.
def create_test_points(name, normal=True):
    if name != 'dragon' and name != 'terrestrial_lidar' and name != 'airborne_lidar' and name != 'box':
        print("name should be dragon, terrestrial_lidar, airborne_lidar or box")
    if name == 'box':
        A1, A2 = create_test_points_box()
        return A1, A2

    A1, A2 = np.array(read_xyz('Data/{}1.xyz'.format(name))), np.array(read_xyz('Data/{}2.xyz'.format(name)))

    if normal:
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(A1)
        vis_pcd.estimate_normals()
        A1_normal = np.asarray(vis_pcd.normals)

        vis_pcd2 = o3d.geometry.PointCloud()
        vis_pcd2.points = o3d.utility.Vector3dVector(A2)
        vis_pcd2.estimate_normals()
        A2_normal = np.asarray(vis_pcd2.normals)
        return A1, A2, A1_normal, A2_normal
    return A1, A2

# Return distance between point and points.
def calc_distances(point, points):
    return ((point - points)**2).sum(axis=1)

# Find the points that spread out the point cloud (for our proposed improved sampling techique).
def find_spread_out_points(A1, k):
    points_to_return = []
    random_i  =np.random.randint(len(A1))
    points_to_return.append(random_i)
    distances = calc_distances(A1[points_to_return[0]], A1)
    for i in range(1, k):
        point = np.argmax(distances)
        points_to_return.append(point)
        distances = np.minimum(distances, calc_distances(A1[point], A1))
    return points_to_return

# Function that adds points (Gaussian) to pointcloud.
def add_noise(pcd, mu, sigma, num_points):
    noisy_pcd = np.copy(pcd)
    noise = np.random.normal(mu, sigma, size=noisy_pcd.shape)
    noisy_pcd += noise
    noisy_indices = random.sample(range(0, len(noisy_pcd)-1), num_points)
    noisy_points = noisy_pcd[noisy_indices]
    pcd = np.concatenate((pcd, noisy_points), axis=0)
    return pcd

def icp(A1, A2, epsilon, sample_size, closest_point_search='KDTree', sampling_method='c', noise_sigma=None, sample_target=False, max_distance=None):
    """
    ### This function computes a translation between two 3D sets of points using
    ### the iterative closest point algorithm.
    ### Input:
    ###      A1: Base/source
    ###      A2: Target
    ###      epsilon: Threshold to know when ICT has converged. If difference in 
    ###               RMS-error is less then epsilon stop itterating.
    ###      closest_point_search: Method to search for closest points (KDTree is fastest)
    ###      sampling_method: Method to sample points from A1 (source/base) 'a' is using 
    ###      all points; 'b' is uniform sub-sampling; 'c' is random sub-sampling in each 
    ###      iteration; 'd'  is using the normals of the pointclouds to sample form more informatively; 
    ###      and 'e' is sub-sampling more from informative regions by using K-means
    ###      noise_sigma: sigma used for adding gaussian noise. If sigma = None, no noise is added.
    ###      sample_target: Set for true if you want to also sample from target otherwise all
    ###      target points are taken into account as a potential closest point to source point
    ###      max_distance: Used to discard point in closest point seleciton that are further away then max_distance
    ### Returns:
    ###      R: rotation matrix
    ###      t: translation vector 
    ###      RMS_list: list with each RMS for every iteration (used for plotting)
    ###      time_list: list with time logged at each iteration (used for plotting)
    """      
    ###### 0. Adding noise
    if noise_sigma:
        mu= 0
        sigma= noise_sigma
        percent_noise = 0.1
        num_points = round(len(A1) * percent_noise)
        A1 = add_noise(A1, mu, sigma, num_points)
        A2 = add_noise(A2, mu, sigma, num_points)
    
    # Compute normals of point clouds
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(A1)
    vis_pcd.estimate_normals()
    A1_normal = np.asarray(vis_pcd.normals)

    vis_pcd2 = o3d.geometry.PointCloud()
    vis_pcd2.points = o3d.utility.Vector3dVector(A2)
    vis_pcd2.estimate_normals()
    A2_normal = np.asarray(vis_pcd2.normals)

    # if closest_point_search == 'KDTree':
    #     kd_tree = sklearn.neighbors.KDTree(A2) # Compute KD-tree

    tic = time.perf_counter() # Set timer
    RMS_list = [] # List used for plotting RMS against iterations
    time_list = [] 

    A1_copy = np.copy(A1) # Copy A1 as to preserve original base A1

    if sampling_method == 'a':
        sample_size, sample_each_itter, sample_target = len(A1) - 1, False, False
        sample_indices_A1 = random.sample(range(0, len(A1_copy)-1), sample_size) # Uniform sampling
    
    elif sampling_method == 'b':
        if sample_target == True:
            sample_indices_A2 = random.sample(range(0, len(A2)-1), sample_size)
        sample_indices_A1, sample_each_itter = random.sample(range(0, len(A1_copy)-1), sample_size), False
    
    elif sampling_method == 'c':
        sample_each_itter = True
    
    elif sampling_method == 'd':
        kmeans = MiniBatchKMeans(n_clusters=100, random_state=0).fit(A1_normal)
        bins_dict = {}
        for i, label in enumerate(kmeans.labels_):
            if label not in bins_dict:
                bins_dict[label] = [i]
            else:
                bins_dict[label].append(i)
        rest = sample_size%100
        sample_indices_A1 = []
        if rest != 0:
            random_buckets = random.sample(range(0, len(bins_dict)-1), int(sample_size-rest)/100)
        for i in bins_dict:
            Bin = bins_dict[i]
            if rest != 0 and i in random_buckets:
                idx = random.sample(range(0, len(Bin)-1), int((sample_size-rest)/100)+1)
            else:
                idx = random.sample(range(0, len(Bin)-1), int((sample_size-rest)/100))
            sample_indices_A1.append(np.array(Bin)[idx].squeeze())

        sample_indices_A1 = np.hstack(sample_indices_A1)
        sample_each_itter = False
        if sample_target == True:
            print("Sampling from target not possible fot set sampling method. Setting sample_target to False")
    
    elif sampling_method == 'e':
        sample_indices_A1 = find_spread_out_points(A1, sample_size)
        if sample_target == True:
            sample_indices_A2 = find_spread_out_points(A2, sample_size)
        sample_each_itter = False
    
    else:
        print("Not a valid sampling method. Use method a, b, c, d or e.")

    if closest_point_search == 'KDTree':
        if sample_each_itter == False and sample_target == True:
            kd_tree = sklearn.neighbors.KDTree(A2[sample_indices_A2]) # Compute KD-tree over samples of A2
        else:
            kd_tree = sklearn.neighbors.KDTree(A2) # Compute KD-tree over A2
    

    rms_prev = 9999 # Set prev_rms
    itter = 0
    while True:

        ##### Step 1: Sample
        if sample_each_itter == True:
            sample_indices_A1 = random.sample(range(0, len(A1_copy)-1), sample_size) # Uniform sampling
        A1_samples = A1_copy[sample_indices_A1]

        if sample_target == True:
            if sample_each_itter == True:
                sample_indices_A2 = random.sample(range(0, len(A2)-1), sample_size) # Uniform sampling
            A2_samples = A2[sample_indices_A2]
            kd_tree = sklearn.neighbors.KDTree(A2_samples) # Create new kd_tree

        #### Step 2: Finds closest points for samples
        if closest_point_search == 'KDTree': # KDTree
            distances, points = kd_tree.query(A1_samples, 1)
            distances, points = distances.squeeze(), points.squeeze()
        else: # Brute Force
            if sample_target == False:
                distances, points = brute_force_closest_point(A1_samples, A2, sampling_method)
            else:
                distances, points = brute_force_closest_point(A1_samples, A2_samples, sampling_method)

        if max_distance != None:
            to_delete = np.where(distances >= max_distance)[0]
            A1_samples = np.delete(A1_samples, to_delete, axis=0)
            points = np.delete(points, to_delete, axis=0)
            temp = np.delete(distances, to_delete, axis=0)
            distances = temp

        if sample_target == False:
            closest_points_A2 = A2[points]
        else:
            closest_points_A2 = A2_samples[points]

        weights = 1 - (distances / np.max(distances))

        ##### Step 3: Compute transformation
        R, t = get_transform(A1_samples, closest_points_A2, weights)

        ##### Step 4: Update source
        A1_copy = np.dot(A1_copy, R.T) + t

        ##### Step 5: Check error
        rms_current = rms(A1_samples, closest_points_A2)
        RMS_list.append(rms_current)

        toc = time.perf_counter()
        time_current = toc-tic
        time_list.append(time_current)

        ##### Step 6: Check if RMS has converged, if so compute and return final transormation
        if abs(rms_prev - rms_current) <= epsilon:
            R, t = get_transform(A1, A1_copy)
            toc = time.perf_counter()
            # print("Number of Itterations needed: {} \nNumber of seconds needed: {} \nFinal RMS: {} \nR: {} \nt: {}\n".format(itter, round(toc-tic, 2), rms_current, R, t))
            print("Number of Iterations needed: {} \nNumber of seconds needed: {:.3f} \nFinal RMS: {:.4f}\n".format(itter, toc-tic, rms_current))
            return R, t, RMS_list, time_list
        rms_prev = rms_current # Update rms_prev
        itter += 1





