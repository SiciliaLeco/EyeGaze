from utils import *
from LeNet import *
import torch

def get_2D_vector(pose, gaze):
    pose2d = []
    gaze2d = []
    for i in np.arange(0, len(pose), 1):
        pose2d.append(pose3D_to_2D(pose[i]))
        gaze2d.append(gaze3D_to_2D(gaze[i]))
    poses = np.array(pose2d)
    gazes = np.array(gaze2d)
    return poses, gazes

def batch_process(j, batch, img, pose, gaze):
    a = torch.randn(batch, 1, 36, 60)
    b = torch.randn(batch,2)
    c = torch.randn(batch,2)
    for i in range(batch):
        a[i, 0] = torch.tensor(img[j * batch + i])
        b[i] = torch.tensor(pose[j * batch + i])
        c[i] = torch.tensor(gaze[j * batch + i])
    return a, b, c


train_gaze, train_image, train_pose, test_gaze, test_image, test_pose = collect_data_from_mat()

train_pose2D, train_gaze2D = get_2D_vector(train_pose, train_gaze)
test_pose2D, test_gaze2D = get_2D_vector(test_pose, test_gaze)

ltrain = len(train_gaze)
ltest = len(test_gaze)
print("training dataset size:", len(train_gaze))
print("test dataset size:", len(test_gaze))


### training process ###
GazeCNN = Model()

optimizer = torch.optim.Adam(GazeCNN.parameters(), lr=0.0001)
criterion = torch.nn.SmoothL1Loss(reduction='mean') #default size_average=True --> /batch_size



for epoch in range(5):
    batch = 10
    for i in tqdm(range(int(ltrain/batch))):
        img, gaze, pose = batch_process(i, batch, train_image, train_pose2D, train_gaze2D)
        gaze_pred_2D = GazeCNN(img, gaze)

        loss = criterion(gaze_pred_2D, gaze)
        loss.backward()
        optimizer.step()

    tloss = 0
    for j in tqdm(range(int(ltest/batch))):
        timg, tgaze, tpose = batch_process(j, batch, train_image, train_pose2D, train_gaze2D)
        tgaze_pred_2D = GazeCNN(timg, tgaze)

        tloss += criterion(tgaze_pred_2D, tgaze)
    print("epoch", epoch, "average loss on test dataset:", tloss/(int(ltest/batch)))
