import datetime
from io import *
import time
import scipy.io as sio
import numpy as np
import os
import torch
from torch import nn,optim
#from first_unet import *
from unet import *
from torch.utils.data import Dataset, DataLoader
import torch
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from scipy.integrate import cumtrapz
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import pywt
import pickle


#---------------------------------------------------------------------#
class nmse(nn.Module):            
    def __init__(self):
        super(nmse, self).__init__()
        
    #def forward(self, x, y):
    def forward(self, x, y):
        a = torch.sum((torch.sub(y,x))**2,dim=2)/torch.sum((y)**2,dim=2)
        #a = torch.mean(torch.sqrt(torch.sum((torch.sub(y,x))**2,dim=2)))/torch.mean(torch.sqrt(torch.sum((y)**2,dim=2)))
        return a
#---------------------------------------------------------------------#




#---------------------------------------------------------------------#
# 导入数据
#---------------------------------------------------------------------#



# 指定.npy文件的路径
#f = r'C:\Users\Dw\Desktop\npy\input_data.npy'

# 使用numpy.load加载数据
# loaded_data = np.load(f)
# print("数据结构",loaded_data.shape)

#
# #---------------------------------------------------------------------#
# # 读取数据
# #---------------------------------------------------------------------#
#
# train_ratio = 0.6  # 训练集比例
# val_ratio = 0.2    # 验证集比例
# test_ratio = 0.2   # 测试集比例
#
# total_samples = loaded_data.shape[0]
#
# # 计算划分后的样本数量
# num_train_samples = int(total_samples * train_ratio)
# num_val_samples = int(total_samples * val_ratio)
# num_test_samples = total_samples - num_train_samples - num_val_samples
#
# # 随机打乱数据集
# np.random.shuffle(loaded_data)
#
# # 划分数据集
# x_train = loaded_data[:num_train_samples, :]  # 前 num_train_samples 行作为训练集
# x_val = loaded_data[num_train_samples:num_train_samples + num_val_samples, :]  # 接下来的 num_val_samples 行作为验证集
# x_test = loaded_data[num_train_samples + num_val_samples:, :]  # 剩余的行作为测试集
#
# x_train = np.array(x_train, dtype=np.float32)
# x_val = np.array(x_val, dtype=np.float32)
# x_test = np.array(x_test, dtype=np.float32)
#
#
# x_train = torch.from_numpy(x_train)
# x_val = torch.from_numpy(x_val)
# x_test = torch.from_numpy(x_test)
#
#
#
# # 假设 y_data 存储在 'C:\Users\Dw\Desktop\npy\input_data.npy' 文件中
# y_data = np.load(r'C:\Users\Dw\Desktop\npy\output_data.npy')
#
# # 接下来，根据你的需要划分数据集，可以使用之前提到的示例代码来划分 y_data
# # 例如，按照训练、验证和测试集比例来划分 y_data
#
# train_ratio = 0.6  # 训练集比例
# val_ratio = 0.2    # 验证集比例
# test_ratio = 0.2   # 测试集比例
#
# total_samples_y = y_data.shape[0]
#
# # 计算划分后的样本数量
# num_train_samples_y = int(total_samples_y * train_ratio)
# num_val_samples_y = int(total_samples_y * val_ratio)
# num_test_samples_y = total_samples_y - num_train_samples_y - num_val_samples_y
#
# # 随机打乱 y 轴标签数据集
# np.random.shuffle(y_data)
#
# # 划分 y 轴标签数据集
# y_train = y_data[:num_train_samples_y, :]  # 前 num_train_samples_y 行作为训练集
# y_val = y_data[num_train_samples_y:num_train_samples_y + num_val_samples_y, :]  # 接下来的 num_val_samples_y 行作为验证集
# y_test = y_data[num_train_samples_y + num_val_samples_y:, :]  # 剩余的行作为测试集
#
# y_train = np.array(y_train, dtype=np.float32)
# y_val = np.array(y_val, dtype=np.float32)
# y_test = np.array(y_test, dtype=np.float32)
#
# y_train = torch.from_numpy(y_train)
# y_val = torch.from_numpy(y_val)
# y_test = torch.from_numpy(y_test)


# f = r'C:\Users\Dw\Desktop\unet\npy\output_data_complete.npy'
data1 = np.load(r'C:\Users\Dw\Desktop\unet\npy\output_data_complete.npy')
data2 = np.load(r'C:\Users\Dw\Desktop\unet\npy\output_data_complete2.npy')
data3 = np.load(r'C:\Users\Dw\Desktop\unet\npy\output_data_complete3.npy')
# 使用numpy.load加载数据
loaded_data = np.concatenate((data1, data2,data3), axis=0)
# loaded_data = np.load(f)
print("数据结构", loaded_data.shape)

# 假设 loaded_data 是你的数据，它的维度为 (175, 20, 1024)

train_ratio = 0.6  # 训练集比例
val_ratio = 0.2    # 验证集比例
test_ratio = 0.2   # 测试集比例

total_samples = loaded_data.shape[0]

# 计算划分后的样本数量
num_train_samples = int(total_samples * train_ratio)
num_val_samples = int(total_samples * val_ratio)
num_test_samples = total_samples - num_train_samples - num_val_samples

# 随机打乱数据集
np.random.shuffle(loaded_data)

# 划分数据集
x_train = loaded_data[:num_train_samples, :, :]  # 前 num_train_samples 行作为训练集
x_val = loaded_data[num_train_samples:num_train_samples + num_val_samples, :, :]  # 接下来的 num_val_samples 行作为验证集
x_test = loaded_data[num_train_samples + num_val_samples:, :, :]  # 剩余的行作为测试集

x_train = np.array(x_train, dtype=np.float32)
x_val = np.array(x_val, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)

# 将数据转换为 PyTorch 张量
x_train = torch.from_numpy(x_train)
x_val = torch.from_numpy(x_val)
x_test = torch.from_numpy(x_test)


# y轴
# y_data = np.load(r'C:\Users\Dw\Desktop\unet\npy\input_data_complete.npy')
data4 = np.load(r'C:\Users\Dw\Desktop\unet\npy\input_data_complete.npy')
data5 = np.load(r'C:\Users\Dw\Desktop\unet\npy\input_data_complete2.npy')
data6 = np.load(r'C:\Users\Dw\Desktop\unet\npy\input_data_complete3.npy')
# 使用numpy.load加载数据
y_data = np.concatenate((data4, data5,data6), axis=0)

# 接下来，根据你的需要划分数据集，按照竖向数组划分 y_data
# 例如，按照训练、验证和测试集比例来划分 y_data

# 假设 y_data 是你的数据，它的维度为 (175, 20, 1024)

# 假设 y_data 是你的数据，它的维度为 (175, 20, 1024)

train_ratio = 0.6  # 训练集比例
val_ratio = 0.2    # 验证集比例
test_ratio = 0.2   # 测试集比例

total_samples_y = y_data.shape[0]

# 计算划分后的样本数量
num_train_samples_y = int(total_samples_y * train_ratio)
num_val_samples_y = int(total_samples_y * val_ratio)
num_test_samples_y = total_samples_y - num_train_samples_y - num_val_samples_y

# 随机打乱 y_data 的竖向数组（按第一个维度打乱）
np.random.shuffle(y_data)

# 划分 y_data 的竖向数组
y_train = y_data[:num_train_samples_y, :, :]  # 前 num_train_samples_y 行作为训练集
y_val = y_data[num_train_samples_y:num_train_samples_y + num_val_samples_y, :, :]  # 接下来的 num_val_samples_y 行作为验证集
y_test = y_data[num_train_samples_y + num_val_samples_y:, :, :]  # 剩余的行作为测试集

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# 将数据转换为 PyTorch 张量
y_train = torch.from_numpy(y_train)
y_val = torch.from_numpy(y_val)
y_test = torch.from_numpy(y_test)



print('x_train.shape',x_train.shape,'x_val.shape',x_val.shape,'x_test.shape',x_test.shape)
print('y_train.shape',y_train.shape,'y_val.shape',y_val.shape,'y_test.shape',y_test.shape)


#---------------------------------------------------------------------#
#----------------------使用数据加载方法--------------------------------#
#---------------------------------------------------------------------#





class MyDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self,x_train,y_train):
        self.x = x_train
        self.y = y_train
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

# #---------------------------------------------------------------------#
# 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
train_dataloader = MyDataset(x_train,y_train)
train_loader = DataLoader(dataset=train_dataloader,
                           batch_size=48,
                           pin_memory=True,
                           #num_workers=1,
                           shuffle=False)
#---------------------------------------------------------------------#
# 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
val_dataloader = MyDataset(x_val, y_val)#[return:# (tensor(x1),tensor(y1));# (tensor(x2),tensor(y2));# ......

val_loader = DataLoader(dataset=val_dataloader,
                        batch_size=48,
                        pin_memory=True,
                        #num_workers=1,
                        shuffle=False)
#---------------------------------------------------------------------#
# 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
test_dataloader = MyDataset(x_test, y_test)#[return:# (tensor(x1),tensor(y1));# (tensor(x2),tensor(y2));#
test_loader = DataLoader(dataset=test_dataloader,
                           batch_size=48,
                           pin_memory=True,
                           shuffle=False,
                           drop_last=False)


# # #---------------------------------------------------------------------#
# # #利用二阶导数四阶差分法求导数
#n为长度
n = 1024

dt = 0.0004
#dt = 0.0004   #h平方
phi1 = np.concatenate([np.array([0.606840,0.298273,-3.535863,3.749543,-1.118795]), np.zeros([n - 5, ])])
phi2 = np.concatenate([np.array([0.916661,-1.666648,0.499975,0.333347,-0.083336]), np.zeros([n - 5, ])])
temp1 = np.concatenate([1  * np.identity(n - 4), np.zeros([n - 4, 4])], axis=1)
temp2 = np.concatenate([np.zeros([n - 4, 1]), -16 * np.identity(n - 4),np.zeros([n - 4, 3])], axis=1)
temp3 = np.concatenate([np.zeros([n - 4, 2]), 30 * np.identity(n - 4),np.zeros([n - 4, 2])], axis=1)
temp4 = np.concatenate([np.zeros([n - 4, 3]), -16 * np.identity(n - 4),np.zeros([n - 4, 1])], axis=1)
temp5 = np.concatenate([np.zeros([n - 4, 4]), 1 * np.identity(n - 4),np.zeros([n - 4, 0])], axis=1)
phi3 = (temp1 + temp2 + temp3+ temp4+ temp5)*(-1/12)
phi4 = np.concatenate([np.zeros([n - 5, ]), np.array([-0.114754,0.590164,-0.081967,-1.147541,0.754098])])
phi5 = np.concatenate([np.zeros([n - 5, ]), np.array([-0.200981,0.549023,0.558818,-1.960781,1.053921])])
Phi_t =1 / dt* np.concatenate(
        [np.reshape(phi1, [1, phi1.shape[0]]),np.reshape(phi2, [1, phi2.shape[0]]),phi3,
         np.reshape(phi4, [1, phi4.shape[0]]),np.reshape(phi5, [1, phi5.shape[0]])], axis=0)
CF = Phi_t.astype(np.float32)
CF = torch.from_numpy(CF).cuda()


# 创建网络模型
module = UNet()
if torch.cuda.is_available():
    module = module.cuda()
# 损失函数
loss_fn = nn.MSELoss().cuda()
# 优化器 
learning_rate = 1e-4
optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
# 训练的轮数
epoch =500
# 储存路径
work_dir = './UNet'
checkpoints_path = './checkpointsnew'
# 添加tensorboard
writer = SummaryWriter("{}/logs".format(work_dir))
outputs_list = []
def train():
    Loss_train = []
    Loss_val = []


    for i in range(epoch):
        # print("-------epoch  {} -------".format(i+1))
        # 训练步骤
        module.train()
        for step, [batch_x, batch_y] in enumerate(train_loader):
            start_time = time.time()

            global_step = len(train_loader) * i + step

            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            outputs = module(batch_x)
            outputs = outputs-torch.mean(outputs, dim=2, keepdim=True)
            # 将每个张量移到CPU并分离
            if i == 0:
                for tensor in outputs:
                    outputs_list.append(tensor.cpu().detach())
                else:
                    pass
            displacement = My_loss2().cuda()                                         #求导两次
            acc = displacement(outputs,CF)
            loss = loss_fn(acc.flatten(), batch_y.flatten())
            running_loss = loss.item()

            # 优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Train Loss', loss, global_step)

        
            print(
                'Epoch {}/{}, Iter {}/{}. train loss = {:.6f}, time = {:.4f}'.format(
                        i, epoch, step, len(train_loader), loss, time.time() - start_time
                    )
            )
            if (step+ 1) % len(train_loader) == 0:
                Loss_train.append(running_loss)

        # checkpoint
        if (i + 1) % 500 == 0:
            torch.save({
                'epoch': i,
                'model': module.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}_{}.ckpt".format(checkpoints_path, i, str(datetime.datetime.now().strftime('%Y%m%d_%H'))
            ))
        
        # testing
        with torch.no_grad():
            for step, [batch_x,batch_y] in enumerate(val_loader):
                start_time = time.time()

                global_step = len(val_loader) * i + step
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()

                module.eval()
                outputs = module(batch_x)
                outputs = outputs-torch.mean(outputs, dim=2, keepdim=True)
                displacement = My_loss2().cuda()
                acc = displacement(outputs,CF)
                loss2 = loss_fn(acc.flatten(), batch_y.flatten())
                running_loss2 = loss2.item()
                writer.add_scalar('Validation Loss', loss2, global_step)


                print(
                    'Iter {}/{}. test loss = {:.6f}, time = {:.3f}'.format(
                            step, len(val_loader), loss2, time.time() - start_time
                        )
                )
                
                if (step+ 1) % len(val_loader) == 0:
                    Loss_val.append(running_loss2)

        file_address = './predict/lossofnoise.mat'
        sio.savemat(file_address, {'loss_train':np.array(Loss_train),'loss_val':np.array(Loss_val)})



if __name__ == "__main__":
    print(len(train_loader.dataset))
    train()
    with open('outputs_list.pkl', 'wb') as file:
        pickle.dump(outputs_list, file)



 

 





