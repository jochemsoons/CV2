import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def pdump(values, filename, dirname='./plotdata/') :
    pickle.dump(values, open(os.path.join(dirname + filename + '_pdump.pkl'), 'wb'))

def pload(filename, dirname='./plotdata/') :
    file = os.path.join(dirname + filename + '_pdump.pkl')
    if not os.path.isfile(file) :
        raise FileNotFoundError(file + " doesn't exist")
    return pickle.load(open(file, 'rb'))

def plot_iterations_rms(method, dataset, sigma):
    plt.figure()
    if method == 'Brute':
        title = 'brute-force'
    elif method == 'KDTree':
        title = 'kd-tree'
    plt.title("Convergence rate of {} method ({})".format(title, dataset))
    for sample_method in ['a', 'b', 'c', 'e']:
        if sample_method == 'a':
            label = 'Sampling all points'
        elif sample_method == 'b':
            label = 'Uniform sub-sampling'
        elif sample_method == 'c':
            label = 'Random sub-sampling'
        elif sample_method == 'e':
            label = 'Normal-space sampling'
        rms_list = pload(method + '_' + sample_method + '_rms')

        plt.plot(rms_list, label=label)
        plt.xlabel('Iterations')
        plt.ylabel('RMS')
    plt.legend()
    plt.savefig("./plots/" + method + '_' + dataset + '_' + sigma + "_RMS")
