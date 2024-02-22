import torch
import math
from utils import calculate_bispectrum_power_spectrum_efficient
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

FOLDER_ROOT = '/scratch/home/kerencohen2/Git/Bispectrum_Cryo/bs_inversion/baseline_comp'


def read_tensor_from_matlab(file, in_train_main = False):
    if in_train_main:
        x = np.loadtxt(file, delimiter=" ")
        x = torch.tensor(x).unsqueeze(1).unsqueeze(0)
    else:
        x = np.loadtxt(file, delimiter=" ")
        x = torch.tensor(x)
    return x

def read_test_results_from_matlab(test_i, folder):
    
    x_true = read_tensor_from_matlab(os.path.join(folder, 'x_true.csv'))
    data = read_tensor_from_matlab(os.path.join(folder, 'data.csv'))
    shifts = float(np.loadtxt(os.path.join(folder, 'shifts.csv'), delimiter=" "))
    x_est = read_tensor_from_matlab(os.path.join(folder, 'x_est.csv'))
    p_est = float(np.loadtxt(os.path.join(folder, 'p_est.csv'), delimiter=" "))
    rel_error_X = float(np.loadtxt(os.path.join(folder, 'rel_error_X.csv'), delimiter=" "))
    tv_error_p = float(np.loadtxt(os.path.join(folder, 'tv_error_p.csv'), delimiter=" "))

    return x_true, data, shifts, x_est, p_est, rel_error_X, tv_error_p

def read_test_results_from_python(test_i, folder):
    #CHANGE LATER!!!
    x_true = read_tensor_from_matlab(os.path.join(folder, 'x_true.csv'))
    data = read_tensor_from_matlab(os.path.join(folder, 'data.csv'))
    shifts = float(np.loadtxt(os.path.join(folder, 'shifts.csv'), delimiter=" "))
    x_est = read_tensor_from_matlab(os.path.join(folder, 'x_est.csv'))
    p_est = float(np.loadtxt(os.path.join(folder, 'p_est.csv'), delimiter=" "))
    rel_error_X = float(np.loadtxt(os.path.join(folder, 'rel_error_X.csv'), delimiter=" "))
    tv_error_p = float(np.loadtxt(os.path.join(folder, 'tv_error_p.csv'), delimiter=" "))

    return x_true, data, shifts, x_est, p_est, rel_error_X, tv_error_p


def compare_to_baseline_results_per_i(test_i, folder_test, 
                                sample_path_matlab, sample_path_python, 
                                fig_folder=f'figures'):
    
    x_true_b, data_b, shifts_b, x_est_b, p_est_b, \
        rel_error_X_b, tv_error_p_b = \
        read_test_results_from_matlab(test_i, sample_path_matlab)

    x_true, data, shifts, x_est, p_est, \
        rel_error_X, tv_error_p = \
            read_test_results_from_python(test_i, sample_path_python)

    full_fig_folder = os.path.join(folder_test, fig_folder)
    if not os.path.exists(full_fig_folder):
        os.makedirs(full_fig_folder)
    fig_path = os.path.join(full_fig_folder, f'x_all_{test_i}.jpg')

    plt.figure()
    plt.title('Comparison between original signal and its reconstructions')
    plt.plot(x_true, label='org')
    plt.plot(x_est, label='tested')
    plt.plot(x_est_b, label='baseline')
    plt.ylabel('signal')
    plt.xlabel('time')
    plt.legend()
    plt.savefig(fig_path)        
    plt.close()
    
    return rel_error_X, rel_error_X_b

def compare_to_baseline_results(n_runs, folder_test, 
                                folder_matlab, folder_python, 
                                rel_error_file='rel_error_X_Xb.csv'):
    rel_error_X_l = ['rel_error_X']
    rel_error_X_b_l = ['rel_error_X_b']
    
    rel_error_path = os.path.join(folder_test, rel_error_file)
    for i in range(n_runs):
        rel_error_X, rel_error_X_b = \
            compare_to_baseline_results_per_i(i, folder_test, 
                                    os.path.join(folder_matlab, f'sample{i}'), 
                                    os.path.join(folder_python, f'sample{i}'))
        rel_error_X_l.append(rel_error_X)    
        rel_error_X_b_l.append(rel_error_X_b)
        
    
    with open(rel_error_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(rel_error_X_l)  # Header row
        writer.writerow(rel_error_X_b_l)
    # Close the file
    csvfile.close()
    
if __name__ == '__main__':
    root = FOLDER_ROOT
    test_name = 'test_1_sample'
    folder_test = os.path.join(root, test_name)
    folder_matlab = os.path.join(folder_test, 'data_from_matlab')
    folder_python = os.path.join(folder_test, 'data_from_python')
    n_runs_per_test = 2
    compare_to_baseline_results(n_runs_per_test, folder_test, folder_matlab,
                                folder_python)