import h5py
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from supplemental_code import save_obj
from sec_4_2 import *

if __name__ == "__main__":

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
        image = 'obama'
        n_epochs = 3000
        l_alpha, l_delta = 0.1, 0.1
        if image == 'leonardo':
            img = plt.imread('./data/leonardo.jpg')
        elif image == 'obama':
            img = plt.imread('./data/obama.jpg')
        elif image == 'messi':
            img = plt.imread('./data/messi.jpg')
        elif image == 'elon':
            img = plt.imread('./data/elon.jpg')
        else:
            print("Specify correct image. Aborting.")
        latent_parameter_estimation(bfm, img, img_name=image, n_epochs=n_epochs, lambda_alpha=l_alpha, lambda_delta=l_delta)
    
    elif question == '4.2.4':
        image = 'obama'
        if image == 'leonardo':
            img = plt.imread('./data/leonardo.jpg')
        elif image == 'obama':
            img = plt.imread('./data/obama.jpg')
        elif image == 'messi':
            img = plt.imread('./data/messi.jpg')
        elif image == 'elon':
            img = plt.imread('./data/elon.jpg')
        else:
            print("Could not find image, please specify correct image. Aborting.")
            exit()
        try:
            alpha = pload('alpha_' + image)
            delta = pload('delta_' + image)
            omega = pload('omega_' + image)
            t = pload('t_' + image)
            get_texture(bfm, img, image, alpha, delta, omega, t)
        except:
            print("Could not load estimated parameters for {} image. Run 4.2.3 first!".format(image))
        
    elif question == '4.2.5':
        n_epochs = 3000
        l_alpha, l_delta = 0.1, 0.1
        img_frames = []
        landmarks_true_list = []
        print("Loading image frames...")
        for i in range(2, 6):
            img = cv2.imread('./data/frame_{}.png'.format(i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_frames.append(img)
            landmarks_ground_truth = detect_landmark(img)
            landmarks_true_list.append(torch.Tensor(landmarks_ground_truth).t())
        multiple_frames(bfm, img_frames, landmarks_true_list, n_epochs=n_epochs, lambda_alpha=l_alpha, lambda_delta=l_delta)

    elif question == '4.2.6':
        image1 = 'obama'
        image2 = 'messi'
        print("Swapping {} and {} images".format(image1, image2))
        img1 = plt.imread('./data/{}.jpg'.format(image1))
        img2 = plt.imread('./data/{}.jpg'.format(image2))
        face_swap(bfm, img1, img2, source_name=image1, target_name=image2)
