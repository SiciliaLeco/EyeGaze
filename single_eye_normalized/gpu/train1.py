from utils import *
from LeNet1 import *
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
    '''
    :return: a-img, b-pose, c-gaze
    '''
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
cuda_gpu = torch.cuda.is_available()
print("cuda is", cuda_gpu)

GazeCNN = Model()
criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(GazeCNN.parameters(), lr=0.01)

batch = 512
train_range = int(ltrain / batch)
test_range = int(ltest / batch)

loss_list = []

for epoch in range(50):
    for i in tqdm(range(train_range)):
        img, pose, gaze = batch_process(i, batch, train_image, train_pose2D, train_gaze2D)
        gaze_pred_2D = GazeCNN(img)
        loss = criterion(gaze_pred_2D, gaze)
        loss.backward()
        optimizer.step()

    angle_loss=0
    for j in tqdm(range(test_range)):
        timg, tpose, tgaze = batch_process(j, batch, train_image, train_pose2D, train_gaze2D)
        tgaze_pred_2D = GazeCNN(timg)


        angle_loss += mean_angle_loss(tgaze_pred_2D, tgaze)

    print("epoch", epoch, "average loss on test dataset:", angle_loss / test_range)
    loss_list.append(angle_loss/test_range)

print(loss_list)
