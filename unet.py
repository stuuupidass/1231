import torch
from torch import nn
from torch.nn import functional as F
import numpy
import numpy as np
from math import sqrt


# ---------------------------------------------------------------------#
class Conv_1(nn.Module):
    def __init__(self):
        super(Conv_1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=8, stride=2, padding=3),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        return self.layer(x)


# ---------------------------------------------------------------------#
class Conv_2(nn.Module):
    def __init__(self):
        super(Conv_2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),
            nn.LeakyReLU(0.01),
            nn.Dropout1d(0.5)
        )

    def forward(self, x):
        return self.layer(x)


# ---------------------------------------------------------------------#
class Conv_3(nn.Module):
    def __init__(self):
        super(Conv_3, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=3),
            nn.LeakyReLU(0.01),
            nn.Dropout1d(0.5)
        )

    def forward(self, x):
        return self.layer(x)


# ---------------------------------------------------------------------#
# ---------------------------------------------------------------------#
class Conv_4(nn.Module):
    def __init__(self):
        super(Conv_4, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=8, stride=2, padding=3),
            nn.LeakyReLU(0.01),
            nn.Dropout1d(0.5)
        )

    def forward(self, x):
        return self.layer(x)


# ---------------------------------------------------------------------#
# ---------------------------------------------------------------------#
class Conv_5(nn.Module):
    def __init__(self):
        super(Conv_5, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU(0.01),
            nn.Dropout1d(0.5)
        )

    def forward(self, x):
        return self.layer(x)


# ---------------------------------------------------------------------#
class Conv_6(nn.Module):
    def __init__(self):
        super(Conv_6, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU(0.01),
            nn.Dropout1d(0.5)
        )

    def forward(self, x):
        return self.layer(x)


# ---------------------------------------------------------------------#
class Conv_7(nn.Module):
    def __init__(self):
        super(Conv_7, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(128, 20, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        return self.layer(x)


# #---------------------------------------------------------------------#

# ---------------------------------------------------------------------#
class UpSample1(nn.Module):
    def __init__(self):
        super(UpSample1, self).__init__()
        self.layer = nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, feature_map):
        out = self.layer(x)
        return torch.cat((out, feature_map), dim=1)


# ---------------------------------------------------------------------#
class UpSample2(nn.Module):
    def __init__(self):
        super(UpSample2, self).__init__()
        self.layer = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, feature_map):
        out = self.layer(x)
        return torch.cat((out, feature_map), dim=1)


# ---------------------------------------------------------------------#
class UpSample3(nn.Module):
    def __init__(self):
        super(UpSample3, self).__init__()
        self.layer = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, feature_map):
        out = self.layer(x)
        return torch.cat((out, feature_map), dim=1)


# ---------------------------------------------------------------------#
class UpSample4(nn.Module):
    def __init__(self):
        super(UpSample4, self).__init__()
        self.layer = nn.ConvTranspose1d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        out = self.layer(x)
        return out


# ---------------------------------------------------------------------#
# ---------------------------------------------------------------------#
class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 20)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out.permute(0, 2, 1)


# ---------------------------------------------------------------------#
# class Self_Attention(nn.Module):
#     # input : batch_size * seq_len * input_dim
#     # q : batch_size * input_dim * dim_k
#     # k : batch_size * input_dim * dim_k
#     # v : batch_size * input_dim * dim_v
#     def __init__(self, input_dim, dim_k, dim_v):
#         super(Self_Attention, self).__init__()
#         self.q = nn.Linear(input_dim, dim_k)
#         self.k = nn.Linear(input_dim, dim_k)
#         self.v = nn.Linear(input_dim, dim_v)
#         self._norm_fact = 1 / sqrt(dim_k)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         Q = self.q(x)  # Q: batch_size * seq_len * dim_k
#         K = self.k(x)  # K: batch_size * seq_len * dim_k
#         V = self.v(x)  # V: batch_size * seq_len * dim_v
#
#         atten = nn.Softmax(dim=-1)(
#             torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
#
#         output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v
#
#         return output.permute(0, 2, 1)


# ---------------------------------------------------------------------#

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_1()
        self.c2 = Conv_2()
        self.c3 = Conv_3()
        self.c4 = Conv_4()
        # self.sa = Self_Attention(512,512,512)
        self.u1 = UpSample1()
        self.c5 = Conv_5()
        self.u2 = UpSample2()
        self.c6 = Conv_6()
        self.u3 = UpSample3()
        self.c7 = Conv_7()
        self.u4 = UpSample4()
        self.dense = Dense()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(R1)
        R3 = self.c3(R2)
        # R4 = self.sa(self.c4(R3))
        R4 = self.c4(R3)
        O1 = self.c5(self.u1(R4, R3))
        O2 = self.c6(self.u2(O1, R2))
        O3 = self.c7(self.u3(O2, R1))
        out = self.dense(self.u4(O3))
        return out


# #---------------------------------------------------------------------#
# 通过做两次一阶差分，达到二阶差分的效果
class My_loss1(nn.Module):
    def __init__(self):
        super(My_loss1, self).__init__()

    def forward(self, x, a):
        input = x.permute(0, 2, 1)

        # 求得面积
        temp1 = torch.matmul(a, input)

        # 面积相加求和
        temp2 = torch.matmul(a, temp1)

        return temp2.permute(0, 2, 1)


# ---------------------------------------------------------------------#
# #---------------------------------------------------------------------#
# 直接做二阶差分
class My_loss2(nn.Module):
    def __init__(self):
        super(My_loss2, self).__init__()

    def forward(self, x, a):
        input = x.permute(0, 2, 1)

        # 求得面积
        temp1 = torch.matmul(a, input)

        # temp2 = torch.matmul(a, temp1)

        return temp1.permute(0, 2, 1)


# ---------------------------------------------------------------------#
# #---------------------------------------------------------------------#
# 滑动窗口滤波
class Movmean(nn.Module):
    def __init__(self):
        super(Movmean, self).__init__()

    def forward(self, x, a):
        input = x.permute(0, 2, 1)

        # 求得面积
        temp1 = torch.matmul(a, input)

        # temp2 = torch.matmul(a, temp1)

        return temp1.permute(0, 2, 1)


# ---------------------------------------------------------------------#
# ---------------------------------------------------------------------#
class error(nn.Module):
    def __init__(self):
        super(error, self).__init__()

    # def forward(self, x, y):
    def forward(self, x, y):
        a = torch.sqrt(torch.sum((torch.sub(y, x)) ** 2, dim=2)) / torch.sqrt(torch.sum((y) ** 2, dim=2))
        # a = torch.mean(torch.sqrt(torch.sum((torch.sub(y,x))**2,dim=2)))/torch.mean(torch.sqrt(torch.sum((y)**2,dim=2)))
        return a


# ---------------------------------------------------------------------#
class nmse(nn.Module):
    def __init__(self):
        super(nmse, self).__init__()

    # def forward(self, x, y):
    def forward(self, x, y):
        a = torch.sum((torch.sub(y, x)) ** 2, dim=2) / torch.sum((y) ** 2, dim=2)
        # a = torch.mean(torch.sqrt(torch.sum((torch.sub(y,x))**2,dim=2)))/torch.mean(torch.sqrt(torch.sum((y)**2,dim=2)))
        return a
# ---------------------------------------------------------------------#