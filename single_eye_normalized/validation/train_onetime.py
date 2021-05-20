'''
This train_onetime.py is to train for onetime,
which will help to draw a curve on validation loss
and train loss.
'''
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
    '''
    :return: a-img, b-pose, c-gaze
    '''

    a = torch.randn(batch, 1, 36, 60)
    b = torch.randn(batch, 2)
    c = torch.randn(batch, 2)
    for i in range(batch):
        a[i, 0] = torch.tensor(img[j * batch + i])
        b[i] = torch.tensor(pose[j * batch + i])
        c[i] = torch.tensor(gaze[j * batch + i])
    return a, b, c

if __name__ == "__main__":

    raw_gaze, raw_image, raw_pose, raw_index = collect_data_from_mat()
    is_gpu = torch.cuda.is_available()
    t_gaze, t_pose, t_image, v_gaze, v_pose, v_image = get_kfold_data(5, 3, raw_gaze, raw_image, raw_pose)
    t_pose_2D, t_gaze_2D = get_2D_vector(t_pose, t_gaze)
    v_pose_2D, v_gaze_2D = get_2D_vector(v_pose, v_gaze)

    ltrain = len(t_gaze)
    lvaild = len(v_gaze)
    print("training dataset size:", ltrain)
    print("test dataset size:", lvaild)

    GazeCNN = Model()
    optimizer = torch.optim.Adam(GazeCNN.parameters(), lr=0.0001)
    criterion = torch.nn.SmoothL1Loss(reduction="mean")
    if is_gpu:
        criterion = criterion.cuda()

    batch = 128
    train_range = int(ltrain / batch)
    test_range = int(lvaild / batch)

    train_loss_list = []
    valid_loss_list = []
    for epoch in range(100):
        for i in tqdm(range(train_range)):
            img, pose, gaze = batch_process(i, batch, t_image, t_pose_2D, t_gaze_2D)
            if is_gpu:
                GazeCNN = GazeCNN.cuda()
                img = img.cuda()
                pose = pose.cuda()
                gaze = gaze.cuda()

            gaze_pred_2D = GazeCNN(img, pose)

            loss = criterion(gaze_pred_2D, gaze)
            loss.backward()
            optimizer.step()

        train_loss = 0
        for k in tqdm(range(train_range - 1)):
            timg, tpose, tgaze = batch_process(k, batch, t_image, t_pose_2D, t_gaze_2D)
            GazeCNN = GazeCNN.cpu()
            tgaze_pred_2D = GazeCNN(timg, tpose)
            train_loss += mean_angle_loss(tgaze_pred_2D, tgaze)

        train_loss = train_loss / (train_range - 1)
        train_loss_list.append(train_loss)

        ## validation result
        valid_loss = 0
        for j in tqdm(range(test_range - 1)):
            vimg, vpose, vgaze = batch_process(j, batch, v_image, v_pose_2D, v_gaze_2D)
            GazeCNN = GazeCNN.cpu()
            vgaze_pred_2D = GazeCNN(vimg, vpose)
            valid_loss += mean_angle_loss(vgaze_pred_2D, vgaze)

        valid_loss = valid_loss / (test_range - 1)
        valid_loss_list.append(valid_loss)

        print("train_loss, valid_loss = [{},{}]".format(train_loss, valid_loss))

    print("valid loss result:", valid_loss_list)
    print("train loss result:", train_loss_list)


