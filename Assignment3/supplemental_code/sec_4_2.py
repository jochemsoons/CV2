import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from supplemental_code import *
from plot import *

# Convert euclidean coordinates to homogeneous
def e2h(data, use_torch=True):
    if use_torch:
        homogeneous_data = torch.ones((data.shape[0], 4))
        homogeneous_data[:,0:3] = data
    else:
        homogeneous_data = np.ones((data.shape[0], 4))
        homogeneous_data[:,0:3] = data
    return homogeneous_data

# Convert homogeneous coordinates to euclidean
def h2e(data):
    return data[:, :3]

# Convert 3d coordinate to 2d coordinate
def from_3d_to_2d(data):
    z_values = data[:, [-1]]
    data = data / z_values
    return data[:, :-1]

# Computes matrix to rotate around x axis for given degree
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

# Computes matrix to rotate around y axis for given degree
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

# Computes matrix to rotate around z axis for given degree
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

# Computes transformation matrix given degrees to rotate around x, y 
# and z axis and traslation for x, y and z
def transformation_matrix(x_rot=0, y_rot=0, z_rot=0, x=1, y=1, z=1, use_torch=False):
    transformation = z_rot_matrix(z_rot, use_torch) @ y_rot_matrix(y_rot, use_torch) @ x_rot_matrix(x_rot, use_torch)
    transformation[0,3], transformation[1,3], transformation[2,3] = x, y, z
    return transformation

# Compute viewport matrix given left, right, top, bottom
def viewport_matrix(vr, vl, vt, vb):
    V = np.array([  [(vr-vl)/2, 0,          0,      (vr+vl)/2],
                    [0,       (vt-vb)/2,    0,      (vt+vb)/2],
                    [0,        0,           0.5,    0.5],
                    [0,        0,           0,      1]])
    return V    


# Compute perspective projection matrix given left, right, top, bottom, far, near
def perspective_projection_matrix(width, height, n, f, fovy, overwrite=False):
    if not overwrite:
        aspect_ratio = width / height
        t = np.tan(fovy / 2) * n
        b = -t
        r = t * aspect_ratio
        l = -t * aspect_ratio
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

# (question 4.2.1)
def compute_G(bfm, landmarks=None, use_torch=False, alpha=None, delta=None):
    mean_id, pca_id, variance_id, mean_exp, pca_exp, variance_exp, mean_color, triangle_topology = get_data(bfm, landmarks)
    if use_torch:
        mean_id = torch.from_numpy(mean_id)
        pca_id = torch.from_numpy(pca_id)
        mean_exp = torch.from_numpy(mean_exp)
        variance_id = torch.from_numpy(variance_id)
        pca_exp = torch.from_numpy(pca_exp)
        variance_exp = torch.from_numpy(variance_exp)
        G = mean_id + pca_id @ (alpha  * torch.sqrt(variance_id)) + mean_exp + pca_exp @ (delta * torch.sqrt(variance_exp))

    # Compute and save G
    elif landmarks is None:
        # Uniformly sample alpha and delta
        alpha, delta = np.random.uniform(-1, 1, 30), np.random.uniform(-1, 1, 20)
        G = mean_id + (pca_id @ (alpha * np.sqrt(variance_id))) + mean_exp + (pca_exp @ (delta * np.sqrt(variance_exp)))
        save_obj('./results/G_sampled.obj', G, mean_color, triangle_topology.T)
        print("Open G_sampled.obj in Meshlab to view sampled pointcloud.")
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

# (second part question 4.2.2)
def landmarks(bfm):
    file = open('Landmarks68_model2017-1_face12_nomouth.anl', mode='r')
    landmarks_str = file.read()
    file.close()
    landmarks = np.array(landmarks_str.split("\n")).astype(int)
    alpha, delta = np.random.uniform(-1, 1, 30), np.random.uniform(-1, 1, 20)
    G = e2h(compute_G(bfm, landmarks, alpha=alpha, delta=delta), use_torch=False)
    T = transformation_matrix(0, 10, 0, 0, 0, -500)
    V = viewport_matrix(vr=1, vl=0, vt=1, vb=0)
    P = perspective_projection_matrix(width=1, height=1, n=1, f=100, fovy=None, overwrite=True)
    G_trans = G @ T
    G_trans = ((V @ P) @ G_trans.T).T  
    plot_landmarks(G, 'original_landmarks')
    plot_landmarks(G_trans, 'transformed_landmarks')
    print("Open original_landmarks.png and transformed_landmarks.png to get plotted results of original landmarks and transformed landmarks.")
    return 

def latent_parameter_estimation(bfm, img, img_name, n_iters=5000, lambda_alpha=0.1, lambda_delta=0.1, plot=True):
    print("Learning latent parameters for {} image".format(img_name))
    # Visualize facial landmark points on the 2D image plane.
    file = open('Landmarks68_model2017-1_face12_nomouth.anl', mode='r')
    landmarks_str = file.read()
    file.close()
    landmark_indices = np.array(landmarks_str.split("\n")).astype(int)
    landmarks_ground_truth = detect_landmark(img)
    landmarks_true = torch.Tensor(landmarks_ground_truth).t()

    alpha = Variable(torch.FloatTensor(30).uniform_(-1, 1), requires_grad=True)
    delta = Variable(torch.FloatTensor(20).uniform_(-1, 1), requires_grad=True)
    omega = Variable(torch.zeros(3), requires_grad=True)
    t = Variable(torch.FloatTensor([0, 0, -500]), requires_grad=True)

    optimizer = torch.optim.Adam([alpha, delta, omega, t], lr=0.1)

    height, width, _ = np.shape(img)
    fovy = 0.5
    n = 1
    f = 256
    vr = width
    vb = height
    vl, vt = 0, 0
    V = torch.from_numpy(viewport_matrix(vr, vl, vt, vb)).float()
    P = torch.from_numpy(perspective_projection_matrix(width, height, n, f, fovy)).float()

    L_lan_list, L_reg_list, L_fit_list = [], [], []
    for iter in np.arange(n_iters):
        G = e2h(compute_G(bfm, landmarks=landmark_indices, use_torch=True, alpha=alpha, delta=delta)).float()
        T = transformation_matrix(omega[0], omega[1], omega[2], t[0], t[1], t[2], use_torch=True)
        G_trans = T @ G.T
        G_trans = V @ P @ G_trans
        landmarks_pred = G_trans[:2, :] / G_trans[3, :]

        optimizer.zero_grad()
        L_lan = torch.mean((landmarks_pred - landmarks_true).pow(2).sum(dim=0).sqrt())
        L_reg = lambda_alpha * torch.sum(alpha.pow(2)) + lambda_delta * torch.sum(delta.pow(2))
        L_fit = L_lan + L_reg
        L_fit.backward()
        optimizer.step()
        L_lan_list.append(L_lan), L_reg_list.append(L_reg), L_fit_list.append(L_fit)
        if iter % 50 == 0:
            print('Iteration: {}/{}, L_lan: {:.4f}, L_reg: {:.4f}, L_fit:{:.4f}'.format(iter, n_iters, L_lan, L_reg, L_fit))
    if plot:
        # Plot learned landmark matchings.
        plot_learned_matches(landmarks_true, landmarks_pred, img, img_name, n_iters)
        plot_loss_iters(L_lan_list, L_reg_list, L_fit_list, img_name, n_iters)

    pdump(alpha.detach(), 'alpha_' + img_name)
    pdump(delta.detach(), 'delta_' + img_name)
    pdump(omega.detach(), 'omega_' + img_name)
    pdump(t.detach(), 't_' + img_name)
    return landmarks_pred, alpha.detach(), delta.detach(), omega.detach(), t.detach()

def get_texture(bfm, img, img_name, alpha, delta, omega, t, plot=True):
    print("Computing transformed mesh and texture for {}".format(img_name))
    height, width, _ = np.shape(img)
    img_original = np.swapaxes(img, 0, 1)
    n = 1
    f = 256
    vr = width
    vb = height
    vl, vt = 0, 0
    fovy = 0.5
    V = torch.from_numpy(viewport_matrix(vr, vl, vt, vb)).float()
    P = torch.from_numpy(perspective_projection_matrix(width, height, n, f, fovy)).float()

    G = e2h(compute_G(bfm, use_torch=True, alpha=alpha, delta=delta)).t().float()
    T = transformation_matrix(omega[0], omega[1], omega[2], t[0], t[1], t[2], use_torch=True)
    G_trans = V @ P @ T @ G
    G_trans = np.asarray(G_trans[:3, :] / G_trans[3, :]).T

    _, _, _, _, _, _, _, triangle_topology = get_data(bfm)
    texture = np.zeros(G_trans.shape)

    for i, vox in enumerate(G_trans):
        x = vox[0]
        y = vox[1]  
        # Get neighbouring pixels.
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
    # render and save image.
    if plot:
        plot_transformed_pcd(img, G_trans, img_name)
        rendered_img = render(G_trans, texture, triangle_topology.T, H=height, W=width)
        plot_texture(rendered_img, img_name)
    return texture, G_trans, triangle_topology.T

def multiple_frames(bfm, img_frames, landmarks_true_list, n_iters=3000, lambda_alpha=0.1, lambda_delta=0.1, plot=True):
    file = open('Landmarks68_model2017-1_face12_nomouth.anl', mode='r')
    landmarks_str = file.read()
    file.close()
    landmark_indices = np.array(landmarks_str.split("\n")).astype(int)
    M = len(img_frames)
    fovy = 0.5
    alpha = Variable(torch.FloatTensor(M, 30).uniform_(-1, 1), requires_grad=True)
    delta = Variable(torch.FloatTensor(M, 20).uniform_(-1, 1), requires_grad=True)
    omega = Variable(torch.zeros(M, 3), requires_grad=True)
    t = torch.zeros(M, 3)
    t[:, -1] = -500
    t = Variable(t, requires_grad=True)

    optimizer = torch.optim.Adam([alpha, delta, omega, t], lr=0.1)

    height, width, _ = np.shape(img_frames[0])
    n = 1
    f = 256
    vr = width
    vb = height
    vl, vt = 0, 0
    V = torch.from_numpy(viewport_matrix(vr, vl, vt, vb)).float()
    P = torch.from_numpy(perspective_projection_matrix(width, height, n, f, fovy)).float()
    
    landmarks_preds = torch.zeros(M, 2, 68)
    for iter in np.arange(n_iters):
        L_lan, L_reg, L_fit = 0, 0, 0
        for i in range(M):
            G = e2h(compute_G(bfm, landmarks=landmark_indices, use_torch=True, alpha=alpha[i], delta=delta[i])).float()    
            T = transformation_matrix(omega[i][0], omega[i][1], omega[i][2], t[i][0], t[i][1], t[i][2], use_torch=True)
            G_trans = T @ G.T
            G_trans = V @ P @ G_trans
            landmarks_pred = G_trans[:2, :] / G_trans[3, :]
            landmarks_preds[i] = landmarks_pred
            L_lan += torch.mean((landmarks_pred - landmarks_true_list[i]).pow(2).sum(dim=0).sqrt())
            # L_lan += (landmarks_pred - landmarks_true_list[i]).pow(2).sum(dim=0).sqrt().sum() / 68
            L_reg += lambda_alpha * torch.sum(alpha[i].pow(2)) + lambda_delta * torch.sum(delta[i].pow(2))
        L_fit = L_lan + L_reg
        optimizer.zero_grad()
        L_fit.backward()
        optimizer.step()
        if iter % 50 == 0:
            print('iter: {}/{}, L_lan: {:.4f}, L_reg: {:.4f}, L_fit:{:.4f}'.format(iter, n_iters, L_lan, L_reg, L_fit))
    
    for i, img_frame in enumerate(img_frames):
        img_name='frame_' + str(i)
        alpha_ = alpha[i].detach()
        delta_ = delta[i].detach()
        omega_ = omega[i].detach()
        t_ = t[i].detach()
        get_texture(bfm, img_frame, img_name, alpha_, delta_, omega_, t_, plot=plot)

    
def face_swap(bfm, source, target, source_name, target_name):
    _, alpha_s, delta_s, omega_s, t_s = latent_parameter_estimation(bfm, source, source_name, plot=False)
    _, alpha_t, delta_t, omega_t, t_t = latent_parameter_estimation(bfm, target, target_name, plot=False)

    texture_s, G_trans_s, triangles_s = get_texture(bfm, source, source_name, alpha_s, delta_s, omega_s, t_s)
    texture_t, G_trans_t, triangles_t = get_texture(bfm, target, target_name, alpha_t, delta_t, omega_t, t_t)
    
    height_s, width_s, _ = np.shape(source)
    height_t, width_t, _ = np.shape(target)

    print("Rendering swapped faces...")
    swap_s_to_t = render(G_trans_t, texture_s, triangles_t, H=height_t, W=width_t)
    swap_t_to_s = render(G_trans_s, texture_t, triangles_s, H=height_s, W=width_s)
    
    swap_s_to_t = paste_swapped_face(swap_s_to_t, target)
    swap_t_to_s = paste_swapped_face(swap_t_to_s, source)

    plot_swapped_face(swap_s_to_t, source_name, target_name)    
    plot_swapped_face(swap_t_to_s, target_name, source_name)

def paste_swapped_face(swapped_face, target):
    target = target.copy()
    non_black_pixels_mask = np.any(swapped_face != [0, 0, 0], axis=-1) 
    target[non_black_pixels_mask] = swapped_face[non_black_pixels_mask] * 255
    return target
    
