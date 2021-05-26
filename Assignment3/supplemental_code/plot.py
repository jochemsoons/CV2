import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_landmarks(G, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.plot(G[:, 0], G[:, 1], 'o')
    for i, xy in enumerate(zip(G[:, 0], G[:, 1])):
        ax.annotate(str(i), xy=xy, textcoords='data')
    if name == 'original_landmarks':
        plt.title('Original landmarks')
    elif name == 'transformed_landmarks':
        plt.title('Transformed landmarks')
    plt.tight_layout()
    plt.savefig('./results/{}.png'.format(name))
    plt.close()

def plot_learned_matches(landmarks_true, landmarks_pred, img, img_name, n_iters):
    plt.figure()
    plt.imshow(img)
    plt.scatter(landmarks_true[0, :], landmarks_true[1, :], label='detected landmarks')
    landmarks_preds_x, landmarks_pred_y = landmarks_pred.detach()[0, :], landmarks_pred.detach()[1, :]
    plt.scatter(landmarks_preds_x, landmarks_pred_y, label='predicted landmarks')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./results/learned_landmarks_{}_{}.png'.format(img_name, n_iters))
    plt.close()

def plot_loss_iters(L_lan_list, L_reg_list, L_fit_list, img_name, n_iters):
    x = np.arange(0, n_iters)
    plt.figure()
    plt.plot(x, L_lan_list, label='$\mathcal{L}_{lan}$')
    plt.plot(x, L_reg_list, label='$\mathcal{L}_{reg}$')
    plt.plot(x, L_fit_list, label='$\mathcal{L}_{fit}$')
    plt.title('Loss throughout optimization iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/loss_plot_{}_{}'.format(img_name, n_iters))
    plt.close()

def plot_transformed_pcd(img, G_trans, img_name):
    plt.figure()
    plt.imshow(img)
    plt.scatter(G_trans[:, 0], G_trans[:, 1], G_trans[:, 2], marker='.')
    plt.tight_layout()
    plt.savefig('./results/point_cloud_transformed_{}.png'.format(img_name))
    plt.close()

def plot_texture(rendered_img, img_name):
    plt.figure()
    plt.imshow(rendered_img)
    plt.tight_layout()
    plt.savefig('./results/texture_{}.png'.format(img_name))
    plt.close()

def plot_swapped_face(swap_img, img_1, img_2):
    plt.figure()
    plt.imshow(swap_img)
    plt.tight_layout()
    plt.savefig('./results/swap_{}_{}.png'.format(img_1, img_2))
    plt.close()