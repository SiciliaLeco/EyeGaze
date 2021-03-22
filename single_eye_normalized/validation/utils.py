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


def data_split_validation(data_dict:dict):
    '''
    split into two part: validation and train
    :return:
    '''
    t_gaze, t_image, t_pose, v_gaze, v_image, v_pose=[],[],[],[],[],[]
    validation = ["p00", "p01"] # set as validation set
    for key, val in tqdm(data_dict.items()):
        for v in val:
            fgaze, fimage, fpose = read_eye_data(v)
            if key in validation: # add to validation data
                if  len(v_gaze) == 0:
                    v_gaze = fgaze
                    v_pose = fpose
                    v_image = fimage
                else:
                    v_gaze = np.append(v_gaze, fgaze, axis = 0)
                    v_pose = np.append(v_pose, fpose, axis = 0)
                    v_image = np.append(v_image, fimage, axis = 0)
            else: # add to train data
                if len(t_gaze) == 0:
                    t_gaze = fgaze
                    t_pose = fpose
                    t_image = fimage
                else:
                    t_gaze = np.append(t_gaze, fgaze, axis = 0)
                    t_pose = np.append(t_pose, fpose, axis = 0)
                    t_image = np.append(t_image, fimage, axis = 0)

    return t_gaze, t_image, t_pose, v_gaze, v_image, v_pose




def collect_data_from_mat():
    '''
    collect data from annotation part
    :param path: path of normalized data
    :return:  list of index, image, pose, gaze
    '''
    mat_files = glob.glob(path + '/Normalized/**/*.mat', recursive = True)
    mat_files.sort()
    gaze = list()
    image = list()
    index = list()
    pose = list()

    ## read data
    for matfile in mat_files:
        pnum = matfile.split('/')[-2]  # pxx
        pday = matfile.split('/')[-1].split('.')[0] # day0x
        if data_dict.__contains__(pnum) == False:
            data_dict[pnum] = []
        data_dict[pnum].append(matfile)

    return data_split_validation(data_dict)



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
        p_x, p_y, p_z = (pred[i][j] for j in range(3))
        t_x, t_y, t_z = (truth[i][j] for j in range(3))
        angles = (p_x * t_x + p_y * t_y + p_z * t_z)/math.sqrt(p_x**2+p_y**2+p_z**2) * math.sqrt(t_x**2+t_y**2+t_z**2)
        ans += math.acos(angles) * 180 / np.pi
    return ans / len(pred)


t_gaze, t_image, t_pose, v_gaze, v_image, v_pose = collect_data_from_mat()

print("training dataset size:", len(t_gaze))
print("test dataset size:", len(v_gaze))