import glob
from tqdm import tqdm
import math
from scipy.io import loadmat

path = "/Users/liqilin/PycharmProjects/untitled/EyeGaze/single_eye_normalized"

def read_eye_data(mat, label):
    mat_data = loadmat(mat)
    right_info = mat_data['data'][label][0, 0]
    gaze = right_info['gaze'][0, 0]
    image = right_info['image'][0, 0]
    pose = right_info['pose'][0, 0]
    return gaze, image, pose

def calc_angle(gaze1, gaze2):
    angle = 0
    for i in range(3):
        angle += gaze1[i] * gaze2[i]
    s1 = math.sqrt(gaze1[1] **2 + gaze1[2]**2 + gaze1[0] ** 2)
    s2 = math.sqrt(gaze2[1]**2 + gaze2[2]**2 + gaze2[0] ** 2)
    return angle / (s1 * s2)

def collect_data_from_mat():
    mat_files = glob.glob(path+'/Normalized/**/*.mat', recursive = True)
    for matfile in tqdm(mat_files[:1]):
        rgaze, rimage, rpose = read_eye_data(matfile, "right")
        lgaze, limage, lpose = read_eye_data(matfile, "left")

        for i in range(len(rgaze)):
            print("left:", rgaze[i])
            print("right:", lgaze[i])
collect_data_from_mat()