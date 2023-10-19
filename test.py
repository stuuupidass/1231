# import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from torch.utils.data import DataLoader
import os
import datetime
import sys
from first_unet import *
from train import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
# import pywt
import scipy.io as sio

# loss_fn = nn.MSELoss().cuda()
# d_test = d_test.cuda()
# loadckpt = './checkpoints/0.05noiseerror0.2726.ckpt'
loadckpt = './checkpointsnew/model_000499_20231019_00.ckpt'
# loadckpt = './checkpointsnew/model_026999_20230705_22.ckpt'
new_list=[]
def test():
    state_dict = torch.load(loadckpt)
    module.load_state_dict(state_dict['model'])
    module.eval()
    with torch.no_grad():
        for step, [batch_x,batch_y] in enumerate(test_loader):
            start_time = time.time()

            global_step = len(train_loader) * step
            if torch.cuda.is_available(): 
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            outputs = module(batch_x)
            new_list.append(outputs.cpu())
            outputs = outputs-torch.mean(outputs, dim=2, keepdim=True)
            # movingaverage = Movmean().cuda()                                         #滑动窗口滤波
            # outputs = movingaverage(outputs,C)
            # outputs = movingaverage(outputs,C)
            # outputs = movingaverage(outputs,C)                                        #50%noise 平滑三次
            displacement = My_loss2().cuda()
            acc = displacement(outputs,CF)

            # loss = loss_fn(acc, batch_x)/loss_fn(Z2,batch_x)
            # loss = loss_fn(acc.flatten(), batch_y.flatten())/loss_fn(Z2.flatten(), batch_y.flatten())
            loss = loss_fn(acc.flatten(), batch_y.flatten())
            # loss = loss_fn(acc, batch_y)
            print('loss', loss, 100*torch.mean(loss, dim=0))
            # loss_acc = loss_fn(acc, batch_x)
            # outputs = outputs-torch.mean(outputs, dim=2, keepdim=True)
            ###########
            # loss_acc = loss_fn(acc, batch_x)/loss_fn(Z2,batch_x)
            # loss_dis = loss_fn(outputs, batch_y)/loss_fn(Z2,batch_y)
            # loss_all =  loss_fn(outputs, batch_y)/loss_fn(Z2,batch_y)+loss_fn(acc, batch_x)/loss_fn(Z2,batch_x)

            # print('loss_acc',loss_acc,100*torch.mean(loss_acc, dim=0))
            # print('loss_dis',loss_dis,100*torch.mean(loss_dis, dim=0))
            # print('loss_all',loss_all,100*torch.mean(loss_all, dim=0))
            


            # print('loss_acc',loss_acc.reshape(1,64),torch.mean(loss_acc, dim=0))
            # print('loss_dis',loss_dis.reshape(1,64),torch.mean(loss_dis, dim=0))
            # print(
            #     'Iter {}/{}. test loss = {:.3f}, time = {:.3f}'.format(
            #             step, len(train_loader), loss, time.time() - start_time
            #         )
            # )
            #a = wavelet_denoising()
            #temp = outputs.cpu().numpy().reshape(1024)
            
            # plt.subplot(2,1,1)
            # plt.plot(batch_y[0].cpu().numpy().reshape(8192))
            # # plt.plot(batch_x[1].cpu().numpy().reshape(18000))
            # # plt.subplot(2,1,2)
            # # plt.plot(acc[1].cpu().numpy().reshape(18000))
            # plt.subplot(2,1,2)
            # # plt.plot(batch_y[0].cpu().numpy().reshape(18000))
            # plt.plot(acc[0].cpu().numpy().reshape(8192))


#             # denoise = wavelet_denoising(temp)
#             # temp1 = denoise- np.mean(denoise)
#             #temp2 = temp- np.mean(temp)
#             plt.subplot(2,1,1)
#             plt.plot(y_test[0].cpu().numpy().reshape(20480))
#             # plt.plot(batch_x[1].cpu().numpy().reshape(18000))
#             # plt.subplot(2,1,2)
#             # plt.plot(acc[1].cpu().numpy().reshape(18000))
#             plt.subplot(2,1,2)
#             # plt.plot(batch_y[0].cpu().numpy().reshape(18000))
#             plt.plot(outputs[0].cpu().numpy().reshape(20480))
#
#             # plt.subplot(4,1,1)
#             # plt.plot(batch_x[2].cpu().numpy().reshape(1024))
#             # plt.subplot(4,1,2)
#             # plt.plot(batch_y[2].cpu().numpy().reshape(1024))
#             # plt.subplot(4,1,3)
#             # plt.plot(acc[2].cpu().numpy().reshape(1024))
#             # plt.subplot(4,1,4)
#             # plt.plot(outputs[2].cpu().numpy().reshape(1024))
#
#             # plt.subplot(2,1,1)
#             # plt.plot(batch_y[20].cpu().numpy().reshape(1024))
#             # plt.plot(outputs[20].cpu().numpy().reshape(1024))
#             # plt.plot(acc[1].cpu().numpy().reshape(1024))
#             # plt.plot(batch_x[1].cpu().numpy().reshape(1024))
#             plt.title('label & predict of test ')
#             plt.ylabel('acc')
#             #plt.xlabel('point')
#             plt.legend(["real","predict"],loc='upper right')
# #---------------------------------------------------------------------#
#             # plt.subplot(2,1,1)
#             # plt.plot(batch_y[2].cpu().numpy().reshape(1024))
#             # plt.plot(outputs[2].cpu().numpy().reshape(1024))
#             # plt.subplot(2,1,1)
#             # plt.plot(outputs[2].cpu().numpy().reshape(1024))
#             # plt.subplot(2,1,2)
#             #plt.plot(temp1)
#             #plt.plot(outputs.cpu().numpy().reshape(1024))
#             # plt.plot(temp2)
#             # plt.plot(batch_y[10].cpu().numpy().reshape(1024))
#             # plt.plot(outputs[10].cpu().numpy().reshape(1024))
#
#             # plt.plot(outputs[1].cpu().numpy().reshape(1024))
#             # plt.plot(d_test[1].cpu().numpy().reshape(1024))
#             # plt.title('dieplacement & label')
#             # plt.ylabel('dis')
#             # #plt.xlabel('point')
#             # plt.legend(["real","predict"],loc='upper right')
# #---------------------------------------------------------------------#
#
#
#             plt.show()  # 显示图表
#
#
#             file_address = './predict/disearth3noise.mat'
#             sio.savemat(file_address, {'displ_pre':np.array(outputs.cpu()).reshape(48,20480),'displ':np.array(batch_y.cpu()).reshape(48,20480)})
#             # sio.savemat('C:/Users/user/Desktop/模型/Frame/predictdata/0.02new.mat', {'displ':np.array(outputs.cpu()).reshape(64,1024)})


if __name__ == "__main__":
    test()
    print(new_list)

output_array = torch.cat(new_list, dim=0).numpy()
np.save('output_results.npy', output_array)

