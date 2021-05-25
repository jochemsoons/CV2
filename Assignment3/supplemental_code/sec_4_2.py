import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import torch
from torch.autograd import Variable

from supplemental_code import *

# Convert euclidean coordinates to homogeneous
def e2h(data, torch_grad=True):
    if torch_grad:
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
    V = np.array([[(vr-vl)/2, 0,       0,   (vr+vl)/2],
        [0,       (vt-vb)/2, 0,   (vt+vb)/2],
        [0,        0,      0.5, 0.5],
        [0,        0,      0,   1]])
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
    P =  np.array([[(2*n)/(r-l), 0,           (r+l)/(r-l),  0],
                [0,           (2*n)/(t-b), (t+b)/(t-b),  0],
                [0,           0,           -(f+n)/(f-n), -(2*f*n)/(f-n)],
                [0,           0,           -1,           0]])
    return P

# Get required data from bfm. landmarks is, if given, a list of indices
def get_data(bfm, landmarks=None):
    if str(type(landmarks)) != "<class 'NoneType'>":
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
        # return G

    # Compute and save G
    elif landmarks is None:
        # Uniformly sample alpha and delta
        alpha, delta = np.random.uniform(-1, 1, 30), np.random.uniform(-1, 1, 20)
        G = mean_id + (pca_id @ (alpha * np.sqrt(variance_id))) + mean_exp + (pca_exp @ (delta * np.sqrt(variance_exp)))
        save_obj('./results/G.obj', G, mean_color, triangle_topology.T)
        print("Open G.obj in Meshlab to view G.")
    else:
        G = mean_id + (pca_id @ np.sqrt(variance_id)) + mean_exp + (pca_exp @ np.sqrt(variance_exp))
    return G

# Compute point cloud and rotate 10 and -10 degrees (first part question 4.2.2)
def rotate_G_pos_and_neg_10_degrees(bfm):
    G = e2h(compute_G(bfm))
    G_rot_pos_10 = G@transformation_matrix(x_rot=0, y_rot=10, z_rot=0, x=0, y=0, z=0)
    G_rot_neg_10 = G@transformation_matrix(x_rot=0, y_rot=-10, z_rot=0, x=0, y=0, z=0)

    _, _, _, _, _, _, mean_color, triangle_topology = get_data(bfm)

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
    G = e2h(compute_G(bfm, landmarks, alpha=alpha, delta=delta))
    T = transformation_matrix(0, 10, 0, 0, 0, -500)
    G_trans = G @ T
    
    V = viewport_matrix(vr=1, vl=0, vt=1, vb=0)
    n = np.min(G_trans[:, 2]) - 1
    f = np.max(G_trans[:, 2]) + 1

    P = perspective_projection_matrix(width=1, height=1, n=n, f=f, fovy=0.5, overwrite=True)
    G_new = ((V @ P) @ G_trans.T).T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    G = ((V @ P) @ G.T).T
    plt.plot(G[:, 0], G[:, 1], 'o')
    for i, xy in enumerate(zip(G[:, 0], G[:, 1])):
        ax.annotate(str(i), xy=xy, textcoords='data')
    plt.title('Original landmarks')
    plt.savefig('./results/original_landmarks.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.plot(G_new[:, 0], G_new[:, 1], 'o')
    for i, xy in enumerate(zip(G_new[:, 0], G_new[:, 1])):
        ax.annotate(str(i), xy=xy, textcoords='data')
    plt.title('Transformed landmarks')
    plt.savefig('./results/transformed_landmarks.png')

    print("Open landmarks.png to get plotted results of original landmarks and transformed landmarks.")

def latent_parameter_estimation(bfm, n_epochs=5000, image='leonardo'):
    if image == 'leonardo':
        img = cv2.imread('./data/leonardo.jpg')
    elif image == 'obama':
        img = cv2.imread('./data/obama.jpg')
    elif image == 'messi':
        img = cv2.imread('./data/messi.jpg')
    elif image == 'elon':
        img = cv2.imread('./data/elon.jpg')
    else:
        img = image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Visualize facial landmark points on the 2D image plane.
    file = open('Landmarks68_model2017-1_face12_nomouth.anl', mode='r')
    landmarks_str = file.read()
    file.close()
    landmark_indices = np.array(landmarks_str.split("\n")).astype(int)
    landmarks_ground_truth = detect_landmark(img)
    landmarks_true = torch.Tensor(landmarks_ground_truth).t()
    
    lambda_alpha, lambda_delta = 0.1, 0.1
    fovy = 0.5
    alpha = Variable(torch.FloatTensor(30).uniform_(-1, 1), requires_grad=True)
    delta = Variable(torch.FloatTensor(20).uniform_(-1, 1), requires_grad=True)
    omega = Variable(torch.zeros(3), requires_grad=True)
    t = Variable(torch.FloatTensor([0, 0, -500]), requires_grad=True)

    optimizer = torch.optim.Adam([alpha, delta, omega, t], lr=0.1)

    height, width, _ = np.shape(img)
    n = 1
    f = 100
    vr = width
    vb = height
    vl, vt = 0, 0
    V = torch.from_numpy(viewport_matrix(vr, vl, vt, vb)).float()
    P = torch.from_numpy(perspective_projection_matrix(width, height, n, f, fovy)).float()

    for epoch in np.arange(n_epochs):
        G_landmark = e2h(compute_G(bfm, landmarks=landmark_indices, use_torch=True, alpha=alpha, delta=delta))
        G_landmark = G_landmark.t().float()
  
        T = transformation_matrix(omega[0], omega[1], omega[2], t[0], t[1], t[2], use_torch=True)
        out = V @ P @ T @ G_landmark
        pred_landmark = out[:2, :] / out[3, :]

        optimizer.zero_grad()
        L_lan = (pred_landmark - landmarks_true).pow(2).sum(dim=0).sqrt().sum() / 68
        L_reg = lambda_alpha * torch.sum(alpha.pow(2)) + lambda_delta * torch.sum(delta.pow(2))
        L_fit = L_lan + L_reg
        L_fit.backward()
        optimizer.step()      
        print('Epoch: {}/{}, L_lan: {:.4f}, L_reg: {:.4f}, L_fit:{:.4f}'.format(epoch, n_epochs, L_lan, L_reg, L_fit))
    # Plot landmark matchings.
    plt.figure()
    plt.imshow(img)
    plt.scatter(landmarks_ground_truth[:, 0], landmarks_ground_truth[:, 1], label='detected landmarks')
    [pred_landmark_x, pred_landmark_y] = pred_landmark.detach().numpy()[:2, :]
    plt.scatter(pred_landmark_x, pred_landmark_y, label='predicted landmarks')
    plt.legend()
    plt.savefig('results/learned_landmarks_{}_{}.png'.format(image, n_epochs))

    pdump(alpha, 'alpha_' + image)
    pdump(delta, 'delta_' + image)
    pdump(omega, 'omega_' + image)
    pdump(t, 't_' + image)
    return pred_landmark, alpha, delta, omega, t

def texture(bfm, image='leonardo'):
    if image == 'leonardo':
        img = cv2.imread('./data/leonardo.jpg')
    elif image == 'obama':
        img = cv2.imread('./data/obama.jpg')
    elif image == 'messi':
        img = cv2.imread('./data/messi.jpg')
    elif image == 'elon':
        img = cv2.imread('./data/elon.jpg')
    else:
        img = image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = np.shape(img)
    print(img.shape)
    img_original = np.swapaxes(img, 0, 1)
    print(img.shape)
    n = 1
    f = 100
    vr = width
    vb = height
    vl, vt = 0, 0
    fovy = 0.5
    V = torch.from_numpy(viewport_matrix(vr, vl, vt, vb)).float()
    P = torch.from_numpy(perspective_projection_matrix(width, height, n, f, fovy)).float()

    try:
        alpha = pload('alpha_' + image).detach()
        delta = pload('delta_' + image).detach()
        omega = pload('omega_' + image).detach()
        t = pload('t_' + image).detach()
    except:
        print("Could not load estimated parameters for this image. Run 4.2.3 first!")

    G = e2h(compute_G(bfm, use_torch=True, alpha=alpha, delta=delta)).t().float()
    T = transformation_matrix(omega[0], omega[1], omega[2], t[0], t[1], t[2], use_torch=True)
    G_trans = V @ P @ T @ G
    G_trans = np.asarray(G_trans[:3, :] / G_trans[3, :]).T

    plt.figure()
    plt.imshow(img)
    plt.scatter(G_trans[:, 0], G_trans[:, 1], G_trans[:, 2], marker='.')
    plt.savefig('results/point_cloud_transformed_{}.png'.format(image))

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
    rendered_img = render(G_trans, texture, triangle_topology.T, H=height, W=width)
    plt.figure()
    plt.imshow(rendered_img)
    plt.savefig('./results/texture_{}.png'.format(image))
    plt.close()    
    return texture, G_trans

# def multiple_frames(bfm):


