import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from supplemental_code import *
from plot import *

# Convert euclidean coordinates to homogeneous.
def e2h(data, use_torch=True):
    if use_torch:
        homogeneous_data = torch.ones((data.shape[0], 4))
        homogeneous_data[:,0:3] = data
    else:
        homogeneous_data = np.ones((data.shape[0], 4))
        homogeneous_data[:,0:3] = data
    return homogeneous_data

# Convert homogeneous coordinates to euclidean.
def h2e(data):
    return data[:, :3]

# Convert 3d coordinate to 2d coordinate.
def from_3d_to_2d(data):
    z_values = data[:, [-1]]
    data = data / z_values
    return data[:, :-1]

# Computes matrix to rotate around x axis for given degree.
def x_rot_matrix(degrees, use_torch=False):
    theta = math.radians(degrees)
    if use_torch:
        return torch.FloatTensor(
                    [[1, 0,               0,                0],
                     [0, math.cos(theta), -math.sin(theta), 0],
                     [0, math.sin(theta), math.cos(theta),  0],
                     [0, 0,               0,                1]])
    else:
        return np.array([[1, 0,               0,                0],
                        [0, math.cos(theta), -math.sin(theta), 0],
                        [0, math.sin(theta), math.cos(theta),  0],
                        [0, 0,               0,                1]])

# Computes matrix to rotate around y axis for given degree.
def y_rot_matrix(degrees, use_torch=False):
    theta = math.radians(degrees)
    if use_torch:
        return torch.FloatTensor(
                    [[math.cos(theta), 0, math.sin(theta),   0],
                     [0,                 1, 0,               0],
                     [-math.sin(theta),  0, math.cos(theta), 0],
                     [0,                 0, 0,               1]])
    else:
        return np.array([[math.cos(theta), 0, math.sin(theta),   0],
                        [0,                 1, 0,               0],
                        [-math.sin(theta),  0, math.cos(theta), 0],
                        [0,                 0, 0,               1]])

# Computes matrix to rotate around z axis for given degree.
def z_rot_matrix(degrees, use_torch=False):
    theta = math.radians(degrees)
    if use_torch:
        return torch.FloatTensor(
                    [[math.cos(theta), -math.sin(theta), 0,  0],
                     [math.sin(theta),  math.cos(theta),  0, 0],
                     [0,                0,                1, 0],
                     [0,                0,                0, 1]])
    else:
        return np.array([[math.cos(theta), -math.sin(theta), 0,  0],
                        [math.sin(theta),  math.cos(theta),  0, 0],
                        [0,                0,                1, 0],
                        [0,                0,                0, 1]])

# Computes transformation matrix given degrees to rotate around x, y and z axis and translation for x, y and z.
def transformation_matrix(x_rot=0, y_rot=0, z_rot=0, x=1, y=1, z=1, use_torch=False):
    transformation = z_rot_matrix(z_rot, use_torch) @ y_rot_matrix(y_rot, use_torch) @ x_rot_matrix(x_rot, use_torch)
    transformation[0,3], transformation[1,3], transformation[2,3] = x, y, z
    return transformation

# Compute viewport matrix given right, left, top, bottom.
def viewport_matrix(vr, vl, vt, vb):
    V = np.array([  [(vr-vl)/2, 0,          0,      (vr+vl)/2],
                    [0,       (vt-vb)/2,    0,      (vt+vb)/2],
                    [0,        0,           0.5,    0.5],
                    [0,        0,           0,      1]])
    return V    


# Compute perspective projection matrix given width, height, n, f and fovy.
def perspective_projection_matrix(width, height, n, f, fovy, overwrite=False):
    # Normal computation of parameters.
    if not overwrite:
        aspect_ratio = width / height
        t = np.tan(fovy / 2) * n
        b = -t
        r = t * aspect_ratio
        l = -t * aspect_ratio
    # Hardcoded parameters for the transformation in 4.2.2. (part 2).
    else:
        r = width
        l = 0
        t = height
        b = 0 
    P =  np.array([[(2*n)/(r-l), 0,         (r+l)/(r-l),    0],
                [0,           (2*n)/(t-b),  (t+b)/(t-b),    0],
                [0,           0,            -(f+n)/(f-n),   -(2*f*n)/(f-n)],
                [0,           0,            -1,             0]])
    return P

# Get required data from bfm. landmarks is, if given, a list of indices
def get_data(bfm, landmarks=None):
    if landmarks is not None:
        mean_id = np.reshape(np.asarray(bfm['shape/model/mean'], dtype=np.float32), (-1, 3))[landmarks]
        pca_id = np.reshape(np.asarray(bfm['shape/model/pcaBasis'][:,:30], dtype=np.float32), (-1, 3, 30))[landmarks]
        mean_exp = np.reshape(np.asarray(bfm['expression/model/mean'], dtype=np.float32), (-1, 3))[landmarks]
        pca_exp = np.reshape(np.asarray(bfm['expression/model/pcaBasis'][:,:20], dtype=np.float32), (-1, 3, 20))[landmarks]
        mean_color = np.reshape(np.asarray(bfm['color/model/mean'], dtype=np.float32), (-1, 3))[landmarks]
    else:
        mean_id = np.reshape(np.asarray(bfm['shape/model/mean'], dtype=np.float32), (-1, 3))
        pca_id = np.reshape(np.asarray(bfm['shape/model/pcaBasis'][:,:30], dtype=np.float32), (-1, 3, 30))
        mean_exp = np.reshape(np.asarray(bfm['expression/model/mean'], dtype=np.float32), (-1, 3))
        pca_exp = np.reshape(np.asarray(bfm['expression/model/pcaBasis'][:,:20], dtype=np.float32), (-1, 3, 20))
        mean_color = np.reshape(np.asarray(bfm['color/model/mean'], dtype=np.float32), (-1, 3))
    
    triangle_topology = np.asarray(bfm['shape/representer/cells'], dtype=np.int32)
    variance_id = np.asarray(bfm['shape/model/pcaVariance'][:30], dtype=np.float32)
    variance_exp = np.asarray(bfm['expression/model/pcaVariance'][:20], dtype=np.float32)

    return mean_id, pca_id, variance_id, mean_exp, pca_exp, variance_exp, mean_color, triangle_topology

# Function to compute point cloud matrix G.
def compute_G(bfm, landmarks=None, use_torch=False, alpha=None, delta=None):
    # Extract data.
    mean_id, pca_id, variance_id, mean_exp, pca_exp, variance_exp, mean_color, triangle_topology = get_data(bfm, landmarks)
    if use_torch:
        mean_id = torch.from_numpy(mean_id)
        pca_id = torch.from_numpy(pca_id)
        mean_exp = torch.from_numpy(mean_exp)
        variance_id = torch.from_numpy(variance_id)
        pca_exp = torch.from_numpy(pca_exp)
        variance_exp = torch.from_numpy(variance_exp)
        G = mean_id + pca_id @ (alpha  * torch.sqrt(variance_id)) + mean_exp + pca_exp @ (delta * torch.sqrt(variance_exp))
    # Compute and save G as object (question 4.2.1)
    elif landmarks is None:
        # Uniformly sample alpha and delta.
        alpha, delta = np.random.uniform(-1, 1, 30), np.random.uniform(-1, 1, 20)
        G = mean_id + (pca_id @ (alpha * np.sqrt(variance_id))) + mean_exp + (pca_exp @ (delta * np.sqrt(variance_exp)))
        save_obj('./results/G_sampled.obj', G, mean_color, triangle_topology.T)
        print("Open G_sampled.obj in Meshlab to view sampled pointcloud.")
    # Compute G given alpha and delta.
    else:
        G = mean_id + (pca_id @ (alpha * np.sqrt(variance_id))) + mean_exp + (pca_exp @ (delta * np.sqrt(variance_exp)))    
    return G

# Compute point cloud and rotate 10 and -10 degrees (first part question 4.2.2)
def rotate_G_pos_and_neg_10_degrees(bfm):
    G = e2h(compute_G(bfm), use_torch=False)
    G_rot_pos_10 = G @ transformation_matrix(x_rot=0, y_rot=10, z_rot=0, x=0, y=0, z=0)
    G_rot_neg_10 = G @ transformation_matrix(x_rot=0, y_rot=-10, z_rot=0, x=0, y=0, z=0)
    _, _, _, _, _, _, mean_color, triangle_topology = get_data(bfm)
    save_obj('./results/G_original.obj', h2e(G), mean_color, triangle_topology.T)
    save_obj('./results/G_rotated_pos_10_deg.obj', h2e(G_rot_pos_10), mean_color, triangle_topology.T)
    save_obj('./results/G_rotatied_neg_10_deg.obj', h2e(G_rot_neg_10), mean_color, triangle_topology.T)
    print("Open G_rotated_pos_10_deg.obj and G_rotated_neg_10_deg.obj in Meshlab to view rotated G.")

# Compute translated and rotated landmarks (second part question 4.2.2).
def landmarks(bfm):
    # Extract vertex index information.
    file = open('Landmarks68_model2017-1_face12_nomouth.anl', mode='r')
    landmarks_str = file.read()
    file.close()
    landmarks = np.array(landmarks_str.split("\n")).astype(int)
    # Sample alpha and delta uniformly.
    alpha, delta = np.random.uniform(-1, 1, 30), np.random.uniform(-1, 1, 20)
    # Compute G, T, V and P.
    G = e2h(compute_G(bfm, landmarks, alpha=alpha, delta=delta), use_torch=False)
    T = transformation_matrix(0, 10, 0, 0, 0, -500)
    V = viewport_matrix(vr=1, vl=0, vt=1, vb=0)
    P = perspective_projection_matrix(width=1, height=1, n=1, f=100, fovy=None, overwrite=True)
    # Compute translated G.
    G_trans = G @ T
    G_trans = ((V @ P) @ G_trans.T).T  
    # Plot results.
    plot_landmarks(G, 'original_landmarks')
    plot_landmarks(G_trans, 'transformed_landmarks')
    print("Open original_landmarks.png and transformed_landmarks.png to get plotted results of original landmarks and transformed landmarks.")
    return 

# Function to estimate latent parameters alpha, delta, omega and t through gradient descent.
def latent_parameter_estimation(bfm, img, img_name, n_iters=5000, lambda_alpha=0.1, lambda_delta=0.1, plot=True):
    print("Learning latent parameters for {} image".format(img_name))
    # Extract vertex index information.
    file = open('Landmarks68_model2017-1_face12_nomouth.anl', mode='r')
    landmarks_str = file.read()
    file.close()
    landmark_indices = np.array(landmarks_str.split("\n")).astype(int)
    # Obtain ground truth landmarks through the provided detect() function.
    landmarks_ground_truth = detect_landmark(img)
    landmarks_true = torch.Tensor(landmarks_ground_truth).t()
    # Plot the detected landmarks.
    if plot:
        plot_learned_matches(landmarks_true, landmarks_pred=None, img=img, img_name=img_name, n_iters=0)
    
    # Initialize the to-be-learned variables as torch Variables.
    alpha = Variable(torch.FloatTensor(30).uniform_(-1, 1), requires_grad=True)
    delta = Variable(torch.FloatTensor(20).uniform_(-1, 1), requires_grad=True)
    omega = Variable(torch.zeros(3), requires_grad=True)
    t = Variable(torch.FloatTensor([0, 0, -500]), requires_grad=True)
    
    # Initialize Adam optimizer.
    optimizer = torch.optim.Adam([alpha, delta, omega, t], lr=0.1)

    # Obtain V and P matrices.
    height, width, _ = np.shape(img)
    fovy = 0.5
    n = 1
    f = 2000
    vr = width
    vb = height
    vl, vt = 0, 0
    V = torch.from_numpy(viewport_matrix(vr, vl, vt, vb)).float()
    P = torch.from_numpy(perspective_projection_matrix(width, height, n, f, fovy)).float()

    # Perform gradient descent.
    L_lan_list, L_reg_list, L_fit_list = [], [], []
    L_fit_prev = np.inf
    for iter in np.arange(n_iters):
        # Step 1: compute G given current alpha and delta.
        G = e2h(compute_G(bfm, landmarks=landmark_indices, use_torch=True, alpha=alpha, delta=delta)).float()
        # Step 2: transform G with T, V and P matrices to get predicted landmarks.
        T = transformation_matrix(omega[0], omega[1], omega[2], t[0], t[1], t[2], use_torch=True)
        G_trans = T @ G.T
        G_trans = V @ P @ G_trans
        landmarks_pred = G_trans[:2, :] / G_trans[3, :]
        # Step 3: reset all gradients.
        optimizer.zero_grad()
        # Step 4: compute landmark loss and regularization loss.
        L_lan = torch.mean((landmarks_pred - landmarks_true).pow(2).sum(dim=0).sqrt())
        L_reg = lambda_alpha * torch.sum(alpha.pow(2)) + lambda_delta * torch.sum(delta.pow(2))
        # Step 5: obtain overall loss and perform backpropagation on variables.
        L_fit = L_lan + L_reg
        L_fit.backward()
        optimizer.step()
        L_lan_list.append(L_lan), L_reg_list.append(L_reg), L_fit_list.append(L_fit)
        # Plot detected and predicted landmarks in first iteration.
        if iter == 0:
            if plot:
                plot_learned_matches(landmarks_true, landmarks_pred, img, img_name, iter+1)
        # At every 50th iteration...
        if iter % 50 == 0:
            # Check for convergence and apply early stopping.
            if L_fit >= L_fit_prev - 0.005:
                print("Detected convergence: applying early stopping.")
                break
            else:
                L_fit_prev = L_fit
            print('Iteration: {}/{}, L_lan: {:.4f}, L_reg: {:.4f}, L_fit:{:.4f}'.format(iter, n_iters, L_lan, L_reg, L_fit))
    if plot:
        # Plot learned landmark predictions over ground truth landmarks.
        plot_learned_matches(landmarks_true, landmarks_pred, img, img_name, iter+1)
        # Plot loss against iterations.
        plot_loss_iters(L_lan_list, L_reg_list, L_fit_list, img_name, iter+1)
    # Dump variables so that we may retrieve them later.
    pdump(alpha.detach(), 'alpha_' + img_name)
    pdump(delta.detach(), 'delta_' + img_name)
    pdump(omega.detach(), 'omega_' + img_name)
    pdump(t.detach(), 't_' + img_name)
    return landmarks_pred, alpha.detach(), delta.detach(), omega.detach(), t.detach()

# Function to obtain transformed mesh and texture given estimated parameters (question 4.2.4).
def get_texture(bfm, img, img_name, alpha, delta, omega, t, plot=True):
    print("Computing transformed mesh and texture for {}".format(img_name))
    height, width, _ = np.shape(img)
    # Swap x and y axes for convenience purposes.
    img_original = np.swapaxes(img, 0, 1)
    # Obtain V and P matrices.
    n = 1
    f = 2000
    vr = width
    vb = height
    vl, vt = 0, 0
    fovy = 0.5
    V = torch.from_numpy(viewport_matrix(vr, vl, vt, vb)).float()
    P = torch.from_numpy(perspective_projection_matrix(width, height, n, f, fovy)).float()
    # Compute the transformed G (mesh).
    G = e2h(compute_G(bfm, use_torch=True, alpha=alpha, delta=delta)).t().float()
    T = transformation_matrix(omega[0], omega[1], omega[2], t[0], t[1], t[2], use_torch=True)
    G_trans = V @ P @ T @ G
    G_trans = np.asarray(G_trans[:3, :] / G_trans[3, :]).T
    # Load the triangle topology.
    _, _, _, _, _, _, _, triangle_topology = get_data(bfm)
    # Initialize the texture with zeros.
    texture = np.zeros(G_trans.shape)
    # Loop over every voxel in the transformed mesh.
    for i, vox in enumerate(G_trans):
        # Get the current x and y coordinates.
        x = vox[0]
        y = vox[1]  
        # Get the neighbouring pixels.
        x1, x2 = int(np.floor(x)), int(np.ceil(x))
        y1, y2 = int(np.ceil(y)), int(np.floor(y))
           
        # Get RGB values of neighbours from original image.
        Q_11 = img_original[x1][y1]
        Q_12 = img_original[x1][y2]
        Q_21 = img_original[x2][y1]
        Q_22 = img_original[x2][y2]

        # If denominator equals zero...
        denom = ((x2 - x1) * (y2 - y1))
        if denom == 0:
            # Set texture to average gray color.
            texture[i] = np.array([128, 128, 128])
        # Otherwise...
        else:
            # Fill in the R,G,B channels through the interpolation formula.
            for c in range(3):
                Q_matrix = np.array([[Q_11[c], Q_12[c]], [Q_21[c], Q_22[c]]])                
                rgb_val = 1 / denom * np.array([x2 - x, x - x1]) @ Q_matrix @ np.array([y2 - y, y - y1])   
                texture[i][c] = rgb_val
    # Normalize values between 0 and 1.
    texture = texture / 255
    if plot:
        # Plot the transformed mesh
        plot_transformed_pcd(img, G_trans, img_name)
        # Render and plot the texture image.
        rendered_img = render(G_trans, texture, triangle_topology.T, H=height, W=width)
        plot_texture(rendered_img, img_name)
    return texture, G_trans, triangle_topology.T

# Function to perform the latent parameter estimation on a series of frames (question 4.2.5).
def multiple_frames(bfm, img_frames, landmarks_true_list, n_iters=5000, lambda_alpha=0.1, lambda_delta=0.1, plot=True):
    # Extract vertex index information.
    file = open('Landmarks68_model2017-1_face12_nomouth.anl', mode='r')
    landmarks_str = file.read()
    file.close()
    landmark_indices = np.array(landmarks_str.split("\n")).astype(int)
    
    # M is the number of frames we have.
    M = len(img_frames)
    # Initialize the to-be-learned variables as torch Variables.
    alpha = Variable(torch.FloatTensor(M, 30).uniform_(-1, 1), requires_grad=True)
    delta = Variable(torch.FloatTensor(M, 20).uniform_(-1, 1), requires_grad=True)
    omega = Variable(torch.zeros(M, 3), requires_grad=True)
    t = torch.zeros(M, 3)
    t[:, -1] = -500
    t = Variable(t, requires_grad=True)

    # Initialize Adam optimizer.
    optimizer = torch.optim.Adam([alpha, delta, omega, t], lr=0.1)

    # Obtain V and P matrices.
    height, width, _ = np.shape(img_frames[0])
    n = 1
    f = 2000
    fovy = 0.5
    vr = width
    vb = height
    vl, vt = 0, 0
    V = torch.from_numpy(viewport_matrix(vr, vl, vt, vb)).float()
    P = torch.from_numpy(perspective_projection_matrix(width, height, n, f, fovy)).float()
    
    # Perform gradient descent over the M images.
    landmarks_preds = torch.zeros(M, 2, 68)
    L_fit_prev = np.inf
    for iter in np.arange(n_iters):
        L_lan, L_reg, L_fit = 0, 0, 0
        # At every epoch loop over all M images.
        for i in range(M):
            # Step 1: compute G given current alpha and delta.
            G = e2h(compute_G(bfm, landmarks=landmark_indices, use_torch=True, alpha=alpha[i], delta=delta[i])).float()  
            # Step 2: transform G with T, V and P matrices to get predicted landmarks.
            T = transformation_matrix(omega[i][0], omega[i][1], omega[i][2], t[i][0], t[i][1], t[i][2], use_torch=True)
            G_trans = T @ G.T
            G_trans = V @ P @ G_trans
            landmarks_pred = G_trans[:2, :] / G_trans[3, :]
            landmarks_preds[i] = landmarks_pred
            # Step 3: compute landmark loss and regularization loss.
            L_lan += torch.mean((landmarks_pred - landmarks_true_list[i]).pow(2).sum(dim=0).sqrt())
            L_reg += lambda_alpha * torch.sum(alpha[i].pow(2)) + lambda_delta * torch.sum(delta[i].pow(2))
        # Step 4: obtain overall loss and perform backpropagation on variables.
        L_fit = L_lan + L_reg
        optimizer.zero_grad()
        L_fit.backward()
        optimizer.step()
        # At every 50th iteration...
        if iter % 50 == 0:
            # Check for convergence and apply early stopping.
            if L_fit >= L_fit_prev - 0.005:
                print("Detected convergence: applying early stopping.")
                break
            else:
                L_fit_prev = L_fit
            print('iter: {}/{}, L_lan: {:.4f}, L_reg: {:.4f}, L_fit:{:.4f}'.format(iter, n_iters, L_lan, L_reg, L_fit))
    # Loop over the image frames.
    for i, img_frame in enumerate(img_frames):
        # Get the alpha, delta, omega and t for the image frame.
        img_name='frame_' + str(i)
        alpha_ = alpha[i].detach()
        delta_ = delta[i].detach()
        omega_ = omega[i].detach()
        t_ = t[i].detach()
        # Obtain and plot the transformed mesh and textures.
        get_texture(bfm, img_frame, img_name, alpha_, delta_, omega_, t_, plot=True)

# Function to perform face swapping given two images (source, target).
def face_swap(bfm, source, target, source_name, target_name):
    # Try to load the estimated parameters of source image.
    try:
        alpha_s = pload('alpha_' + source_name)
        delta_s = pload('delta_' + source_name)
        omega_s = pload('omega_' + source_name)
        t_s = pload('t_' + source_name)
    # Otherwise estimate them with our earlier defined function.
    except:
        _, alpha_s, delta_s, omega_s, t_s = latent_parameter_estimation(bfm, source, source_name, plot=False)
    # Try to load the estimated parameters of target image.
    try:
        alpha_t = pload('alpha_' + target_name)
        delta_t = pload('delta_' + target_name)
        omega_t = pload('omega_' + target_name)
        t_t = pload('t_' + target_name)
    # Otherwise estimate them with our earlier defined function.
    except:
        _, alpha_t, delta_t, omega_t, t_t = latent_parameter_estimation(bfm, target, target_name, plot=False)
    
    # Obtain the texture and transformed mesh for the source and target images.
    texture_s, G_trans_s, triangles_s = get_texture(bfm, source, source_name, alpha_s, delta_s, omega_s, t_s, plot=False)
    texture_t, G_trans_t, triangles_t = get_texture(bfm, target, target_name, alpha_t, delta_t, omega_t, t_t, plot=False)
    # Get image shape information for rendering.
    height_s, width_s, _ = np.shape(source)
    height_t, width_t, _ = np.shape(target)

    # Render swapped faces (source to target and target to source).
    print("Rendering swapped faces...")
    swap_s_to_t = render(G_trans_t, texture_s, triangles_s, H=height_t, W=width_t)
    swap_t_to_s = render(G_trans_s, texture_t, triangles_s, H=height_s, W=width_s)
    
    # Paste swapped faces onto original images to get final swapped face image.
    swap_s_to_t = paste_swapped_face(swap_s_to_t, target)
    swap_t_to_s = paste_swapped_face(swap_t_to_s, source)
    # Plot results.
    plot_swapped_face(swap_s_to_t, source_name, target_name)    
    plot_swapped_face(swap_t_to_s, target_name, source_name)

# Helper function that pastes a (cropped) swapped face onto the corresponding target image.
def paste_swapped_face(swapped_face, target):
    target = target.copy()
    # Detect all pixels that are non-black in the face image.
    non_black_pixels_mask = np.any(swapped_face != [0, 0, 0], axis=-1) 
    # Replace these pixels in the target image with the swapped face.
    target[non_black_pixels_mask] = swapped_face[non_black_pixels_mask] * 255
    return target
    
