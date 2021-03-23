'''
Why is it neccessary to covert 3D vector into 2D?
What if we train using 3D poses and gazes?
'''
from utils import *
from LeNet2 import *
import torch

def batch_process(j, batch, img, pose, gaze):
    '''
    :return: a-img, b-pose, c-gaze
    '''
    a = torch.randn(batch, 1, 36, 60)
    b = torch.randn(batch,3)
    c = torch.randn(batch,3)
    for i in range(batch):
        a[i, 0] = torch.tensor(img[j * batch + i])
        b[i] = torch.tensor(pose[j * batch + i])
        c[i] = torch.tensor(gaze[j * batch + i])
    return a, b, c

train_gaze, train_image, train_pose, test_gaze, test_image, test_pose = collect_data_from_mat()

ltrain = len(train_gaze)
ltest = len(test_gaze)
print("training dataset size:", len(train_gaze))
print("test dataset size:", len(test_gaze))


### training process ###
cuda_gpu = torch.cuda.is_available()
GazeCNN = Model()
optimizer = torch.optim.Adam(GazeCNN.parameters(), lr=0.0001)
criterion = torch.nn.SmoothL1Loss(reduction="mean")
batch = 512
train_range = int(ltrain / batch)
test_range = int(ltest / batch)


for epoch in range(1):
    for i in tqdm(range(1)):
        img, pose, gaze = batch_process(i, batch, train_image, train_pose, train_gaze)
        np.array(train_gaze)
        gaze_pred = GazeCNN(img, pose)
        loss = criterion(gaze_pred, gaze)
        loss.backward()
        optimizer.step()

    angle_loss=0
    for j in tqdm(range(1)):
        timg, tpose, tgaze = batch_process(j, batch, train_image, train_pose, train_gaze)
        tgaze_pred = GazeCNN(timg, tpose)
        print(mean_angle_loss(tgaze_pred, tgaze))

    print("epoch", epoch, "average loss on test dataset:", angle_loss / test_range)

