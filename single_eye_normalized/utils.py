from scipy.io import loadmat
import glob
from tqdm import tqdm
import numpy as np
import cv2 as cv


def read_eye_data(mat):
    '''
    read each mat file info
    '''
    mat_data = loadmat(mat)
    right_info = mat_data['data']['right'][0, 0]
    gaze = right_info['gaze'][0, 0]
    image = right_info['image'][0, 0]
    pose = right_info['pose'][0, 0]
    return gaze, image, pose

def collect_data_from_mat():
    '''
    collect data from annotation part
    :return:  list of index, image, pose, gaze
    '''
    mat_files = glob.glob('Normalized/**/*.mat', recursive = True)
    mat_files.sort()
    # dict to store
    index = list()
    # X: image, head_pose
    # y: gaze vector
    # index: pnum, pday
    i = 0
    train_gaze, train_image, train_pose, test_gaze, test_image, test_pose=[],[],[],[],[],[]
    for matfile in tqdm(mat_files):
        pnum = matfile.split('/')[-2]  # pxx
        pday = matfile.split('/')[-1].split('.')[0] # day0x
        index.append(pnum + '/' + pday)

        fgaze, fimage, fpose = read_eye_data(matfile)
        if int(pnum[1:]) < 10:
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

def pose3D_to_2D(pose):
    '''
      pose (a, b, c) is rotation (angle)
      M = Rodrigues((x,y,z))
      Zv = (the third column of M)
      theta = asin(Zv[1])
      phi = atan2(Zv[0], Zv[2])
    '''
    M, _ = cv.Rodrigues(np.array(pose).astype(np.float32))
    vec = M[:, 2]
    phi = np.arctan2(vec[0], vec[2])
    theta = np.arcsin(vec[1])
    return np.array([theta, phi])

def gaze3D_to_2D(gaze):
    '''
      gaze (x, y, z) is direction
      theta = asin(-y)
      phi = atan2(-x, -z)
    '''
    x, y, z = (gaze[i] for i in range(3))
    theta = np.arcsin(-y)
    phi = np.arctan2(-x, -z)
    return np.stack((theta, phi)).T