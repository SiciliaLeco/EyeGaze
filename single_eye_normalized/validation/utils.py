from scipy.io import loadmat
import glob
from tqdm import tqdm
import numpy as np
import cv2 as cv
import torch
import math
from scipy.io import loadmat

path = "/Users/liqilin/PycharmProjects/untitled/EyeGaze/single_eye_normalized"

data_dict = dict()

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
    :param path: path of normalized data
    :return:  list of index, image, pose, gaze
    '''
    mat_files = glob.glob(path + '/Normalized/**/*.mat', recursive = True)
    # mat_files.sort()
    gaze = list()
    image = list()
    index = list()
    pose = list()
    for matfile in tqdm(mat_files[:5]):
        pnum = matfile.split('/')[-2]  # pxx
        pday = matfile.split('/')[-1].split('.')[0] # day0x
        index.append(pnum + '/' + pday)

        fgaze, fimage, fpose = read_eye_data(matfile)

        if gaze == []:
            gaze = fgaze
            image = fimage
            pose = fpose
        else:
            gaze = np.append(gaze, fgaze, axis = 0)
            image = np.append(image, fimage, axis = 0)
            pose = np.append(pose, fpose, axis = 0)

    return gaze, image, pose, index


def get_kfold_data(k, i, gaze, image, pose):
    '''
    implement k-fold validation
    input type = numpy.narray
    output type = numoy.narray
    '''
    fold_size = gaze.shape[0] // k
    start = i * fold_size
    if i != k - 1: # Not the final round
        end = (i + 1) * fold_size
        v_gaze, v_pose, v_image = gaze[start:end], pose[start:end],image[start:end]
        t_gaze = np.concatenate((gaze[0:start], gaze[end:]), axis=0)
        t_pose = np.concatenate((pose[0:start], pose[end:]), axis=0)
        t_image = np.concatenate((image[0:start], image[end:]), axis=0)
    else:
        v_gaze, v_pose, v_image = gaze[start:], pose[start:],image[start:]
        t_gaze, t_pose, t_image = gaze[0:start], pose[0:start],image[0:start]

    return t_gaze, t_pose, t_image, v_gaze, v_pose, v_image


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
    pitch = gaze[0]
    yaw = gaze[1]
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


# gaze, image, pose, index = collect_data_from_mat()
# t_gaze, t_pose, t_image, v_gaze, v_pose, v_image=get_kfold_data(10,0,gaze,image,pose)
