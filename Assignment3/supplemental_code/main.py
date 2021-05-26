import h5py
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from supplemental_code import detect_landmark
from sec_4_2 import *

def main():
    question = str(input("Enter name of question for which results need to be produced. \nQuestions can be 4.2.1, 4.2.2, etc.\n"))
    while question not in ['4.2.1', '4.2.2', '4.2.3', '4.2.4', '4.2.5', '4.2.6']:
        question = str(input("Not a valid question number. Needs to be 4.2.1, 4.2.2, etc.: "))
    # question = '4.2.6'
    bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')
    Path("./results/").mkdir(parents=True, exist_ok=True)

    if question == '4.2.1':
        compute_G(bfm)
    
    elif question == '4.2.2':
        rotate_G_pos_and_neg_10_degrees(bfm)
        landmarks(bfm)
    
    elif question == '4.2.3':
        image = 'floris'
        n_iters = 5000
        l_alpha, l_delta = 0.1, 0.1
        try:
            img = plt.imread('./data/{}.jpg'.format(image))
        except:
            print("Could not find {} image in data folder. Please specify correct image. Aborting.".format(image))
            return
        latent_parameter_estimation(bfm, img, img_name=image, n_iters=n_iters, lambda_alpha=l_alpha, lambda_delta=l_delta)
        
    
    elif question == '4.2.4':
        image = 'messi'
        try:
            img = plt.imread('./data/{}.jpg'.format(image))
        except:
            print("Could not find {} image in data folder. Please specify correct image. Aborting.".format(image))
            return
        try:
            alpha = pload('alpha_' + image)
            delta = pload('delta_' + image)
            omega = pload('omega_' + image)
            t = pload('t_' + image)
        except:
            _, alpha, delta, omega, t = latent_parameter_estimation(bfm, img, image, plot=False)
        get_texture(bfm, img, image, alpha, delta, omega, t)
        
    elif question == '4.2.5':
        n_iters = 5000
        l_alpha, l_delta = 0.1, 0.1
        img_frames = []
        landmarks_true_list = []
        print("Loading image frames...")
        for i in range(5):
            img = cv2.imread('./data/frame_{}.png'.format(i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_frames.append(img)
            landmarks_ground_truth = detect_landmark(img)
            landmarks_true_list.append(torch.Tensor(landmarks_ground_truth).t())
        multiple_frames(bfm, img_frames, landmarks_true_list, n_iters=n_iters, lambda_alpha=l_alpha, lambda_delta=l_delta)

    elif question == '4.2.6':
        image1 = 'floris'
        image2 = 'wouter'
        try:
            print("Swapping {} and {} images".format(image1, image2))
            img1 = plt.imread('./data/{}.jpg'.format(image1))
            img2 = plt.imread('./data/{}.jpg'.format(image2))
        except:
            print("Could not find {} and {} images in data folder. Please specify correct images. Aborting.".format(image1, image2))
            return
        face_swap(bfm, img1, img2, source_name=image1, target_name=image2)

if __name__ == "__main__":
    main()

