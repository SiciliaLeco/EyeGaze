from scipy.io import loadmat
import glob
from tqdm import tqdm
import numpy as np
import cv2 as cv
import torch

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
    yaw = np.arctan2(vec[0], vec[2])
    pitch = np.arcsin(vec[1])
    return np.array([pitch, yaw])


def gaze3D_to_2D(gaze):
    '''
      gaze (x, y, z) is direction
      theta = asin(-y)
      phi = atan2(-x, -z)
    '''
    x, y, z = (gaze[i] for i in range(3))
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.stack((pitch, yaw)).T


def gaze2D_to_3D(gaze):
    '''
    :param gaze: gaze (yaw, pitch) is the rotation angle, type=(list)
    :return: gaze=(x,y,z)
    '''
    yaw = gaze[1]
    pitch = gaze[0]
    x = -np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def mean_angle_loss(pred, truth):
    '''
    :param pred,truth: type=torch.Tensor
    :return:
    '''
    pred = pred.detach().numpy()
    ans = 0
    for i in range(len(pred)):
        p_x, p_y, p_z = gaze2D_to_3D(pred[i])
        t_x, t_y, t_z = gaze2D_to_3D(truth[i])
        angles = p_x * t_x + p_y * t_y + p_z * t_z
        ans += torch.acos(angles) * 180 / np.pi

    return ans / len(pred)