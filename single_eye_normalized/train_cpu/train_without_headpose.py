from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(3600, 500)
        self.fc2 = nn.Linear(500, 2)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        self.apply(initialize_weights)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten
        x = self.fc2(x)
        return x

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
GazeCNN = Model()
optimizer = torch.optim.Adam(GazeCNN.parameters(), lr=0.001)
criterion = torch.nn.SmoothL1Loss(reduction="mean")
batch = 10
train_range = int(ltrain / batch)
test_range = int(ltest / batch)


for epoch in range(10):
    for i in tqdm(range(train_range)):
        img, pose, gaze = batch_process(i, batch, train_image, train_pose2D, train_gaze2D)
        gaze_pred_2D = GazeCNN(img)

        loss = criterion(gaze_pred_2D, gaze)
        loss.retain_grad()
        loss.backward()
        optimizer.step()

    angle_loss=0
    for j in tqdm(range(test_range)):
        timg, tpose, tgaze = batch_process(j, batch, test_image, test_pose2D, test_gaze2D)
        tgaze_pred_2D = GazeCNN(timg)
        angle_loss += mean_angle_loss(tgaze_pred_2D, tgaze)

    print("epoch", epoch, "average loss on test dataset:", angle_loss / test_range)

