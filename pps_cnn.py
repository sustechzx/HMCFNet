# ###################################################### 6通道数据 ##################################################
# EOG,EMG,GSR,PPG
# import warnings
# from torch import nn
#
# import torch.nn.functional as F
# warnings.filterwarnings("ignore")
#
# import torch
#
#
# class cnn_classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv11 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv12 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool3d(kernel_size=2, padding=(0, 1, 1))
#
#         self.conv21 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool3d(kernel_size=2, padding=(0, 0,1), stride=(1,1,2))
#
#         # self.fc_layer = nn.Linear(512, 2)
#
#         # self.dropout_layer = nn.Dropout(p=0.3)
#
#     def forward(self, xb):    # 100,1,6,5,3
#         h1 = self.conv11(xb)  # 100,32,6,5,3,
#         h1 = self.conv12(h1)  # 100,32,6,5,3
#         h1 = self.pool1(h1)   # 100,32,3,3,2
#         h1 = F.relu(h1)       # 100,32,3,3,2
#
#         h2 = self.conv21(h1)  # 100,64,3,3,2
#         h2 = self.conv22(h2)  # 100,64,3,3,2
#         h2 = self.pool2(h2)   # 100,64,2,2,2
#         h2 = F.relu(h2)       # 100,64,2,2,2
#
#         # Before the fully connected layer, we need to flatten the output
#         flatten = h2.view(-1, 64*2*2*2)   # 25,512
#
#         # out = self.fc_layer(flatten)
#
#         return flatten

# ########################################## 4通道数据 ################################
# EMG,GSR,PPG
# import warnings
# from torch import nn
#
# import torch.nn.functional as F
# warnings.filterwarnings("ignore")
#
# import torch
#
#
# class cnn_classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv11 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv12 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool3d(kernel_size=2, padding=(1, 1, 1))
#
#         self.conv21 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool3d(kernel_size=2, padding=(0, 0,1), stride=(1,1,2))
#
#         # self.fc_layer = nn.Linear(512, 2)
#
#         # self.dropout_layer = nn.Dropout(p=0.3)
#
#     def forward(self, xb):    # 100,1,4,5,3
#         h1 = self.conv11(xb)  # 100,32,4,5,3,
#         h1 = self.conv12(h1)  # 100,32,4,5,3
#         h1 = self.pool1(h1)   # 100,32,3,3,2
#         h1 = F.relu(h1)       # 100,32,3,3,2
#
#         h2 = self.conv21(h1)  # 100,64,3,3,2
#         h2 = self.conv22(h2)  # 100,64,3,3,2
#         h2 = self.pool2(h2)   # 100,64,2,2,2
#         h2 = F.relu(h2)       # 100,64,2,2,2
#
#         # Before the fully connected layer, we need to flatten the output
#         flatten = h2.view(-1, 64*2*2*2)   # 25,512
#
#         # out = self.fc_layer(flatten)
#
#         return flatten



# ########################################## 2通道数据 ################################

from torch import nn
import torch
import torch.nn.functional as F

class cnn_classifier(nn.Module):
    def __init__(self):
        super(cnn_classifier,self).__init__()
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, padding=(1, 1, 1), stride=(1, 2, 2))

        self.conv21 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, padding=(0, 0, 1), stride=(1, 1, 2))

        # self.dropout_layer = nn.Dropout(p=0.3)

    def forward(self, xb):
        # print("xb含有nan值：")
        # print(torch.isnan(xb).any())

        h1 = self.conv11(xb)
        h1 = self.conv12(h1)
        h1 = self.pool1(h1)
        h1 = F.relu(h1)

        h2 = self.conv21(h1)
        h2 = self.conv22(h2)
        h2 = self.pool2(h2)
        h2 = F.relu(h2)
        # h2 = F.sigmoid(h2)

        # Before the fully connected layer, we need to flatten the output
        flatten = h2.view(-1, 64*2*2*2)   # (B, 512)

        return flatten