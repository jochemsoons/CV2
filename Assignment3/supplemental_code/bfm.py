import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

from supplemental_code import save_obj
from sec_4_2 import *

if __name__ == "__main__":

    # question = str(input("Enter name of question for which results need to be produced. \nQuestions can be 4.2.1, 4.2.2, etc.\n"))
    # while question not in ['4.2.1', '4.2.2', '4.2.3', '4.2.4']:
    #     question = str(input("Not a valid question number. Needs to be 4.2.1, 4.2.2, etc.: "))
    question = '4.2.4'
    bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')
    Path("./results/").mkdir(parents=True, exist_ok=True)

    if question == '4.2.1':
        compute_G(bfm)
    elif question == '4.2.2':
        rotate_G_pos_and_neg_10_degrees(bfm)
        landmarks(bfm)
    elif question == '4.2.3':
        latent_parameter_estimation(bfm)
    elif question == '4.2.4':
        texture(bfm)