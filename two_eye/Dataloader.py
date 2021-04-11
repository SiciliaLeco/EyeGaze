from scipy.io import loadmat
import glob
from tqdm import tqdm
import numpy as np


path = "/Users/liqilin/PycharmProjects/untitled/EyeGaze/single_eye_normalized"

def read_eye_data(mat, label):
    '''
    read data from each .mat
    :param mat: file name
    :param label: right/ left
    :return: gaze, image, pose
    '''
    mat_data = loadmat(mat)
    right_info = mat_data['data'][label][0, 0]
    gaze = right_info['gaze'][0, 0]
    image = right_info['image'][0, 0]
    pose = right_info['pose'][0, 0]
    return gaze, image, pose

def collect_data_from_mat(label):
    '''
    collect data from annotation part
    :return:  list of index, image, pose, gaze
    '''
    mat_files = glob.glob(path+'/Normalized/**/*.mat', recursive = True)
    mat_files.sort()
    i = 0
    train_gaze, train_image, train_pose, test_gaze, test_image, test_pose=[],[],[],[],[],[]
    for matfile in tqdm(mat_files):
        pnum = matfile.split('/')[-2]  # pxx
        fgaze, fimage, fpose = read_eye_data(matfile, label)
        if int(pnum[1:]) < 7:
            if train_gaze == []:
                train_gaze = fgaze
                train_image = fimage
                train_pose = fpose
            else:
                train_gaze = np.append(train_gaze, fgaze, axis = 0)
                train_image = np.append(train_image, fimage, axis = 0)
                train_pose = np.append(train_pose, fpose, axis = 0)
        else:
            if test_gaze == []:
                test_gaze = fgaze
                test_image = fimage
                test_pose = fpose
            else:
                test_gaze = np.append(test_gaze, fgaze, axis = 0)
                test_image = np.append(test_image, fimage, axis = 0)
                test_pose = np.append(test_pose, fpose, axis = 0)
        i += 1
    return train_gaze, train_image, train_pose, test_gaze, test_image, test_pose