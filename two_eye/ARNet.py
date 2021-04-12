import torch
import torch.nn as nn
import torch.nn.functional as F

############# definition of angle #################

def AngularErr(input,target):
    input = F.normalize(input)
    target = F.normalize(target)
    cosineLoss = nn.CosineSimilarity()
    return cosineLoss(input, target)

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, pred_vec, left_gt, right_gt):
        left_pd = pred_vec[:,:2]
        right_pd = pred_vec[:,2:]
        vall = AngularErr(left_gt, left_pd)
        vall = torch.acos(vall)
        dvall = torch.div(1, vall)
        valr = AngularErr(right_gt, right_pd)
        valr = torch.acos(valr)
        dvalr = torch.div(1, valr)
        sum = torch.add(dvalr, dvall)
        weightl = torch.div(dvall, sum)
        weightr = torch.div(dvalr, sum)
        return torch.add(torch.mul(weightl, vall), torch.mul(weightr, valr))

########### Base-CNN set up ############

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(3600, 500)
        # self.fc2 = nn.Linear(503, 3)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        self.apply(initialize_weights)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True) #flatten
        # x = torch.cat([x, y], dim=1)
        # x = self.fc2(x)
        return x

############### AR-NET set up ################

class ARNet(nn.Module):
    def __init__(self):
        super(ARNet, self).__init__()

        self.bCNN1 = BaseCNN()
        self.bCNN2 = BaseCNN()
        self.fc1 = nn.Linear(1004, 4)

    def forward(self, x1, x2, y1, y2):
        ### x1: left image, x2: right image, y: head pose
        x1 = self.bCNN1(x1)
        x2 = self.bCNN2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, y1], dim=1)
        x = torch.cat([x, y2], dim=1)
        x = self.fc1(x)

        return x