import numpy as np
import cv2 as cv
import torch

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
    z /= norm # all normalized
    return x, y, z


def angle_error(pred, truth1, truth2):
    '''
    :param pred:
    :param truth1:
    :param truth2:
    :return:
    '''
    pred1 = pred[:,:2] # left
    pred2 = pred[:2,:] # right
    ans1 = mean_angle_loss(pred2, truth2)
    ans2 = mean_angle_loss(pred1, truth1)
    if ans2 > ans1:
        return ans1
    else:
        return ans2


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

