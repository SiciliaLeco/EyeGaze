from Dataloader import *
from ARNet import *
from utils import *
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


train_gazel, train_imagel, train_posel, test_gazel, test_imagel, test_posel = collect_data_from_mat("left")
train_gazer, train_imager, train_poser, test_gazer, test_imager, test_poser = collect_data_from_mat("right")

###### transfer to 2D vectors ######

train_pose2Dl, train_gaze2Dl = get_2D_vector(train_posel, train_gazel)
test_pose2Dl, test_gaze2Dl = get_2D_vector(test_posel, test_gazel)
train_pose2Dr, train_gaze2Dr = get_2D_vector(train_poser, train_gazer)
test_pose2Dr, test_gaze2Dr = get_2D_vector(test_poser, test_gazer)

print("training dataset size:", len(train_gazel))
print("test dataset size:", len(test_gazel))

print("training dataset size:", len(train_gazer))
print("test dataset size:", len(test_gazer))

cuda_gpu = torch.cuda.is_available()

GazeNet = ARNet()
optimizer = torch.optim.Adam(GazeNet.parameters(), lr=0.0001)
# criterion = Criterion()
criterion = torch.nn.SmoothL1Loss(reduction="mean")
batch = 10
train_range = int(len(train_gaze2Dl) / batch)
test_range = int(len(test_gaze2Dl) / batch)

for epoch in range(1):
    for i in tqdm(range(2)):
        imgl, posel, gazel = batch_process(i, batch, train_imagel, train_pose2Dl, train_gaze2Dl)
        imgr, poser, gazer = batch_process(i, batch, train_imagel, train_pose2Dl, train_gaze2Dl)
        if cuda_gpu:
            GazeNet = GazeNet.cuda()
            criterion = criterion.cuda()
            imgl = imgl.cuda()
            posel = posel.cuda()
            gazel = gazel.cuda()
            imgr = imgr.cuda()
            poser = poser.cuda()
            gazer = gazer.cuda()
        gaze_pred_2D = GazeNet(imgl, imgr, posel, poser)
        gaze_trut_2D = torch.cat([gazel, gazer], dim=1)
        loss = criterion(gaze_pred_2D, gaze_trut_2D)

        loss.backward(loss.clone().detach())
        optimizer.step()

    valid_loss=0
    for j in tqdm(range(test_range - 1)):
        vimgl, vposel, vgazel = batch_process(j, batch, test_imagel, test_pose2Dl, test_gaze2Dl)
        vimgr, vposer, vgazer = batch_process(j, batch, test_imager, test_pose2Dr, test_gaze2Dr)
        if cuda_gpu:
            GazeCNN = GazeNet.cpu()
        vgaze_pred_2D = GazeNet(vimgl, vimgr, vposel, vposer)
        valid_loss += angle_error(vgaze_pred_2D, vgazel, vgazer)

    print(valid_loss/(test_range-1))
