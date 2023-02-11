
import torch.nn as nn
import torch
import torch.nn.functional as f






# class dsc_block(nn.Module):
#     def __init__(self,in_channel,out_channel):
#         super(dsc_block,self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#
#         self.d_conv = nn.Conv3d(in_channels=in_channel,out_channels=in_channel,kernel_size=5,stride=1,padding=2,groups=in_channel).cuda()  # 深度卷积
#         self.p_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1,groups=1).cuda()  # 点卷积
#
#         self.bn = nn.BatchNorm3d(in_channel).cuda()
#         self.prelu = nn.ELU().cuda()
#
#     def forward(self,x):
#         # print(x.shape)                   # 4,16,96,96,96
#         # print(self.in_channel,self.out_channel)   # 16,32
#         out = self.d_conv(x)      #  分了16组  每组  4，1，96，96，96
#         out = self.bn(out)
#         out = self.prelu(out)
#         out = self.p_conv(out)
#         return out
#
# class se_block(nn.Module):
#     def __init__(self,in_channel,out_channel,r=16,L=4):
#         super(se_block, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.k_3_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, stride=1,padding=2)
#         self.bn = nn.BatchNorm3d(out_channel)
#         self.prelu = nn.PReLU()
#         self.sigmoid = torch.sigmoid
#         self.ave_pooling = nn.AdaptiveAvgPool3d(1)
#         d = max(out_channel // r,L)
#         self.fc1 = nn.Linear(out_channel, d)  # .cuda()
#         self.fc2 = nn.Linear(d, out_channel)  # .cuda()
#     def forward(self,x):
#         out = self.k_3_conv(x)
#         out = self.bn(out)
#         out = self.prelu(out)
#         # print(out.shape)
#         out1d = self.ave_pooling(out)  ##  （BD,C,1*1*1）
#         out1d = torch.flatten(out1d, start_dim=1)
#         # print(out1d.shape)
#         out_mid = self.fc1(out1d)
#         out_mid = self.prelu(out_mid)
#         out_out = self.fc2(out_mid)
#         out_out = self.sigmoid(out_out)
#         out_out = out_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         # print(out_out.shape)
#         out = torch.mul(out, out_out)
#         return out
#
# class se_block1(nn.Module):
#     def __init__(self,in_channel,out_channel,r=16,L=4):
#         super(se_block1, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.k_3_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, stride=1,padding=2)
#         self.bn = nn.BatchNorm3d(out_channel)
#         self.prelu = nn.ELU()
#         self.sigmoid = torch.sigmoid
#         self.softmax = torch.softmax
#         self.ave_pooling = nn.AdaptiveAvgPool3d(1)
#         d = max(out_channel // r,L)
#         self.fc1 = nn.Linear(out_channel, d)  # .cuda()
#         self.fc2 = nn.Linear(d, out_channel)  # .cuda()
#     def forward(self,x):
#         out = self.k_3_conv(x)
#         out = self.bn(out)
#         out = self.prelu(out)
#         # print(out.shape)
#         out1d = self.ave_pooling(out)  ##  （BD,C,1*1*1）
#         out1d = torch.flatten(out1d, start_dim=1)
#         # print(out1d.shape)
#         out_mid = self.fc1(out1d)
#         out_mid = self.prelu(out_mid)
#         out_out = self.fc2(out_mid)
#         out_out = self.prelu(out_out)
#         out_out = self.softmax(out_out,dim=1)
#         out_out = out_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         # print(out_out.shape)
#         out = torch.mul(out, out_out)
#         out = self.bn(out)
#         out = self.prelu(out)
#         return out
#
class mv2_block(nn.Module):  #定义了一个名为 mv2_block 的 3D 的卷积模块
    def __init__(self,in_channel,out_channel):# 该函数定义了模块的初始化操作，并接受两个参数 ，in_channel 和 out_channel，表示该模块的输入通道数和输出通道数
        super(mv2_block,self).__init__() #该语句调用了父类的初始化操作。
        self.in_channel = in_channel     #该语句将输入通道数存储为模块的一个属性。
        self.out_channel = out_channel   #该语句将输出通道数存储为模块的一个属性。


##深度卷积 (Depthwise Convolution) 和点卷积 (Pointwise Convolution) 是卷积神经网络中常用的两种卷积方式。
#深度卷积是对每一个输入通道独立地进行卷积操作，不同的通道不会相互影响，它的目的是为了在保持每个特征图的形状不变的前提下增加特征图的数量，增加网络的表示能力。
# 点卷积则是对每个通道独立进行卷积操作，但是不保持形状不变，目的是对每个特征图增加通道数，使网络学到更多的复杂的特征。点卷积和深度卷积结合可以实现卷积神经网络中的卷积操作，从而增加网络的表示能力。
        # self.d_conv = nn.Conv3d(in_channels=2 * in_channel,out_channels=2 * in_channel,kernel_size=5,stride=1,padding=2,groups=2*in_channel)  # 深度卷积
        self.d_conv = nn.Conv3d(in_channels=2 * in_channel,out_channels=2 * in_channel,kernel_size=3,stride=1,padding=1,groups=2*in_channel)  # 深度卷积，d_conv ：是深度卷积，卷积核大小为 3，步长为 1，填充为 1。
        # self.d_conv = se_block(2*in_channel, 2 * in_channel)  # 深度卷积

        self.p_conv1 = nn.Conv3d(in_channels=in_channel, out_channels=2 * in_channel, kernel_size=1, stride=1,groups=1)  # 点卷积1 p_conv1 ：是点卷积，卷积核大小为 1，步长为 1。
        # self.p_conv1 = se_block(in_channel, 3 * in_channel)  # 点卷积1

        self.p_conv2 = nn.Conv3d(in_channels=2 * in_channel, out_channels=out_channel, kernel_size=1, stride=1,groups=1)  # 点卷积2 p_conv2 ：是点卷积，卷积核大小为 1，步长为 1。
        # self.p_conv2 = se_block(3 * in_channel, out_channel)

        self.prelu = nn.PReLU()   #该语句创建了一个 PReLU 层。激活函数

        self.bn1 = nn.BatchNorm3d(in_channel*2)  #该语句创建了一个批标准化， 两个批标准化层 bn1 和 bn2。
        self.bn2 = nn.BatchNorm3d(out_channel)

    def forward(self,x):     #网络的前向传播过程的代码。
        resres = x           #定义一个变量resres并将输入的张量赋值给它
        mv2res = res_block(self.in_channel, self.out_channel, "pointconv")  # 实例化一个res_block模块，并将该模块赋值给变量mv2res。这个res_block是一个不同的模块
        resres = mv2res(resres)   #这行代码调用了上一行实例化的res_block模块，并将输入的张量传给它，得到了一个输出，并将该输出赋值给变量resres

        outupc = self.p_conv1(x)   #通过第一个点卷积操作得到输出，并将该输出赋值给变量outupc
        # outupc = self.bn1(outupc)
        outupc = self.prelu(outupc)  #通过PReLU激活函数对输入进行处理，并将输出赋值给变量outupc

        out = self.d_conv(outupc)      #  分了16组  每组  4，1，96，96，96   通过深度卷积操作得到输出，并将该输出赋值给变量out
        # out = self.bn1(out)
        out = self.prelu(out)    #通过PReLU激活函数对输入进行处理，并将输出赋值给变量out

        out = self.p_conv2(out)    #通过第二个点卷积操作得到输出，并将该输出
        out = self.bn2(out)
        ### 线性激活函数我没找到  用 y=x代替咯
        out = resres.add(out)
        return out

 ##在这个 forward 函数中，它定义了一个 mv2_block 模块的前向传播过程。

#首先，它调用了函数 res_block，该函数的输入为 mv2_block 的输入通道数（in_channel）和输出通道数（out_channel），以及字符串 "pointconv"。然后将调用 res_block 的结果赋值给变量 resres。

#接下来，它定义了一个点卷积操作，通过调用 self.p_conv1 实现，将输入 x 进行卷积。然后，通过调用 prelu 函数实现对点卷积结果的激活。

#然后，经过一个深度卷积操作，通过调用 self.d_conv 实现。接着，它再进行一次 prelu 激活。

#接下来，再经过一个点卷积操作，通过调用 self.p_conv2 实现。最后，再通过调用 self.bn2 进行一次批归一化。

#最后，它将 resres 和经过上述所有操作后的结果相加，作为最终的输出。
#
# class sk_block(nn.Module):  ###   明天照着这个继续写   https://blog.csdn.net/zahidzqj/article/details/105982058
#     def __init__(self, in_channel, out_channel, M=2, r=48, L=32):  ###   M是分支数，r是降维比率，L是维度下界
#         super(sk_block, self).__init__()
#         self.in_channel = in_channel  ####  我们需要的 输入 要等与  输出
#         self.out_channel = out_channel
#         self.M = M
#         self.r = r
#         self.L = L
#         g = min(in_channel, 16, out_channel)
#         self.k_3_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
#                                   padding=1, groups=g)  # .cuda()
#         self.dilated_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
#                                       padding=2, dilation=2, groups=g)  # .cuda()  # 膨胀卷积
#         # self.dilated_conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, stride=1,#
#         #                               padding=2, dilation=1, groups=g)  # .cuda()  # 膨胀卷积
#         self.dilated_conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,#
#                                       padding=3, dilation=3, groups=g)  # .cuda()  # 膨胀卷积
#         # self.dilated_conv2 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=7, stride=1,
#         #                               padding=3, dilation=1, groups=g)  # .cuda()  # 膨胀卷积
#         self.ave_pooling = nn.AdaptiveAvgPool3d(1)  # .cuda()   #  全局平均池化
#         d = max(out_channel*3 // r, L)
#         self.fc1 = nn.Linear(out_channel*3, d)  # .cuda()
#         self.fc2 = nn.Linear(d, out_channel)  # .cuda()
#         self.softmax = f.softmax
#         self.prelu = nn.ELU()
#         self.bn = nn.BatchNorm3d(out_channel)
#         self.bn1 = nn.BatchNorm1d(d)
#         # self.drop = nn.Dropout3d()
#
#     def forward(self, x):
#         out1 = self.k_3_conv(x)  ##  这里 通道数 变了       (BS,C,SHAPE)
#         out1 = self.bn(out1)
#         out1 = self.prelu(out1)
#         out2 = self.dilated_conv(x)
#         out2 = self.bn(out2)
#         out2 = self.prelu(out2)
#         out3 = self.dilated_conv1(x)#
#         out3 = self.bn(out3)#
#         out3 = self.prelu(out3)#
#         # out4 = self.dilated_conv2(x)
#         # out4 = self.bn(out4)
#         # out4 = self.prelu(out4)
#         out = torch.cat((out1,out2,out3),1)
#         # print("out.shape", out.shape)
#         # out = out1.add(out2)
#         # out = out.add(out3)#
#
#         # out = out.add(out4)
#         out1d = self.ave_pooling(out)  ##  （BD,C,1*1*1）
#         out1d = torch.flatten(out1d, start_dim=1)
#
#         out = self.fc1(out1d)
#         # out = self.bn1(out)
#         out = self.prelu(out)
#
#         outfc1 = self.fc2(out)
#         outfc2 = self.fc2(out)
#         outfc3 = self.fc2(out)#
#         # outfc4 = self.fc2(out)
#
#         #
#         outfc1 = self.prelu(outfc1)
#         # print("outfc1.shape",outfc1.shape)
#         outfc2 = self.prelu(outfc2)
#         outfc3 = self.prelu(outfc3)#
#         # outfc4 = self.prelu(outfc4)
#         outfc = torch.cat((outfc1, outfc2,outfc3), 0)
#         # print(outfc.shape)
#         # outfc = torch.cat((outfc1, outfc2,outfc3,outfc4), 0)
#
#         out = self.softmax(outfc, 1)  #
#         k_3_out = out[0, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
#         dil_out = out[1, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
#         dil_out1 = out[2, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
#         # dil_out2 = out[3, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
#         se1 = torch.mul(k_3_out, out1)  ###   这里两个不同大小的张量要相乘了   先把一个张量扩张一下   再点乘
#         se2 = torch.mul(dil_out, out2)
#         se3 = torch.mul(dil_out1, out3)
#         # se4 = torch.mul(dil_out2, out4)
#         out = se1.add(se2)
#         out = out.add(se3)
#         # out = out.add(se4)
#         return out  # 有正有负，在0附近
#
# class DAMS_block(nn.Module):  ###   明天照着这个继续写   https://blog.csdn.net/zahidzqj/article/details/105982058
#     def __init__(self, in_channel, out_channel):  ###   M是分支数，r是降维比率，L是维度下界
#         super(DAMS_block, self).__init__()
#         self.in_channel = in_channel  ####  我们需要的 输入 要等与  输出
#         self.out_channel = out_channel
#
#         self.k_3_conv = nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1,
#                                   padding=1)
#
#         self.point = nn.Conv3d(in_channels=in_channel*3 , out_channels=out_channel, kernel_size=1, stride=1)
#
#         self.ave_pooling = nn.AdaptiveAvgPool3d(1)  # .cuda()   #  全局平均池化
#         self.ave_pooling1 = nn.AdaptiveAvgPool1d(1)  # .cuda()   #  全局平均池化
#         self.fc0 = nn.Linear(3, 3)  # .cuda()
#         self.fc1 = nn.Linear(out_channel, out_channel)  # .cuda()
#         self.softmax = f.softmax
#         self.prelu = nn.ELU()
#         self.bn = nn.BatchNorm3d(in_channel)
#         self.bn0 = nn.BatchNorm3d(in_channel*3)
#         self.bn1 = nn.BatchNorm3d(out_channel)
#         self.softmax = f.softmax
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x1 = x
#
#         x2 = self.k_3_conv(x)
#         x2 = self.bn(x2)
#         x2 = self.prelu(x2)
#
#
#         x3 = self.k_3_conv(x2)
#         x3 = self.bn(x3)
#         x3 = self.prelu(x3)
#
#
#         x4 = self.k_3_conv(x3)
#         x4 = self.bn(x4)
#         x4 = self.prelu(x4)
#
#
#
#         x11 = self.ave_pooling(x1)  # x11工具人
#         x11 = torch.flatten(x11,1)  # torch.Size([3, 1])
#         x11 = x11.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
#         x11 = self.ave_pooling1(x11)  # 在添加的维度上对所有channel取平均
#
#         x22 = self.ave_pooling(x2)
#         x22 = torch.flatten(x22,1)  # torch.Size([3, 1])
#         x22 = x22.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
#         x22 = self.ave_pooling1(x22)  # 在添加的维度上对所有channel取平均
#
#         x33 = self.ave_pooling(x3)
#         x33 = torch.flatten(x33,1)  # torch.Size([3, 1])
#         x33 = x33.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
#         x33 = self.ave_pooling1(x33)  # 在添加的维度上对所有channel取平均
#
#
#
#
#
#         out1 = torch.cat((x11,x22,x33),1)  # 在添加的维度上进行concat
#         out1 = torch.flatten(out1,1)
#         out1 = self.fc0(out1)
#         out1 = self.prelu(out1)
#         out1 = self.fc0(out1)
#         # out1 = self.sigmoid(out1)
#         # out1 = self.prelu(out1)
#         out1 = self.softmax(out1,1)
#         out1 = out1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 恢复  #torch.Size([channel,1,1,1])
#         # print(out1[:,0,:].unsqueeze(1).shape)
#         # print(x1.shape)
#         x111 = torch.mul(out1[:, 0, :].unsqueeze(1),x1)
#         x222 = torch.mul(out1[:, 1, :].unsqueeze(1),x2)
#         x333 = torch.mul(out1[:, 2, :].unsqueeze(1),x3)
#
#         out0 = torch.cat((x111,x222,x333),1)  # 尺度注意over
#         # # out0 = x111+x222+x333  # 尺度注意over
#         #
#         #
#         # # out0 = torch.cat((x1, x2), 1)  # 尺度注意over
#         # # out0 = x1+x2  # 尺度注意over
#         # out0 = self.bn0(out0)
#         # out0 = self.prelu(out0)
#         # # print(out0.shape)
#         #
#         # # out2 = self.ave_pooling(out0)
#         # # out2 = torch.flatten(out2,1)
#         # # # print(out2.shape)
#         # # out2 = self.fc1(out2)
#         # # out2 = self.prelu(out2)
#         # # out2 = self.fc2(out2)
#         # # out2 = self.prelu(out2)
#         # # out2 = self.softmax(out2, 1)
#         # # out2 = out2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 恢复  #torch.Size([channel,1,1,1])
#         # # out2 = torch.mul(out0,out2)  # 通道注意over
#         # # out2 = self.bn1(out2)
#         # # out2 = self.prelu(out2)
#         # # out = self.point(out2)
#         out = self.point(out0)
#         # out = self.bn1(out)
#         # out = self.prelu(out)
#         #
#         # out1d = self.ave_pooling(out)  ##  （BD,C,1*1*1）
#         # out1d = torch.flatten(out1d, start_dim=1)
#         # # print(out1d.shape)
#         # out_mid = self.fc1(out1d)
#         # out_mid = self.prelu(out_mid)
#         # out_out = self.fc1(out_mid)
#         #
#         # out_out = self.prelu(out_out)
#         # out_out = self.softmax(out_out,1)
#         # out_out = out_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         # # print(out_out.shape)
#         # out = torch.mul(out, out_out)
#
#
#         return out  # 有正有负，在0附近
#
# class r2_block(nn.Module):
#     def __init__(self,in_channel,out_channel,s=4):
#         super(r2_block, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.prelu = nn.ELU()
#         if in_channel==1 and out_channel==16:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=16, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(16)
#             self.slicon = nn.Conv3d(in_channels=4*2,out_channels=4*2,kernel_size=3,stride=1,padding=1)
#             self.bn2 = nn.BatchNorm3d(4*2)
#             self.fcc = 16
#             self.fccbn = nn.BatchNorm3d(16)
#             self.point2 = nn.Conv3d(in_channels=16, out_channels=out_channel, kernel_size=1, stride=1)
#         elif in_channel==16 and out_channel==32:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=32, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(32)
#             self.slicon = nn.Conv3d(in_channels=8*2,out_channels=8*2,kernel_size=3,stride=1,padding=1)
#             self.bn2 = nn.BatchNorm3d(8*2)
#             self.fcc = 32
#             self.fccbn = nn.BatchNorm3d(32)
#             self.point2 = nn.Conv3d(in_channels=32, out_channels=out_channel, kernel_size=1, stride=1)
#         elif in_channel == 32 and out_channel==64:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=64, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(64)
#             self.slicon = nn.Conv3d(in_channels=16*2, out_channels=16*2, kernel_size=3, stride=1, padding=1)
#             self.bn2 = nn.BatchNorm3d(16*2)
#             self.fcc = 64
#             self.fccbn = nn.BatchNorm3d(64)
#             self.point2 = nn.Conv3d(in_channels=64, out_channels=out_channel, kernel_size=1, stride=1)
#         elif in_channel==64 and out_channel==128:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=128, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(128)
#             self.slicon = nn.Conv3d(in_channels=32*2,out_channels=32*2,kernel_size=3,stride=1,padding=1)
#             self.bn2 = nn.BatchNorm3d(32*2)
#             self.fcc = 128
#             self.fccbn = nn.BatchNorm3d(128)
#             self.point2 = nn.Conv3d(in_channels=128, out_channels=out_channel, kernel_size=1, stride=1)
#         elif in_channel==128 and out_channel==256:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=256, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(256)
#             self.slicon = nn.Conv3d(in_channels=64*2,out_channels=64*2,kernel_size=3,stride=1,padding=1)
#             self.bn2 = nn.BatchNorm3d(64*2)
#             self.fcc = 256
#             self.fccbn = nn.BatchNorm3d(256)
#             self.point2 = nn.Conv3d(in_channels=256, out_channels=out_channel, kernel_size=1, stride=1)
#         elif in_channel==256 and out_channel==256:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=256, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(256)
#             self.slicon = nn.Conv3d(in_channels=64*2,out_channels=64*2,kernel_size=3,stride=1,padding=1)
#             self.bn2 = nn.BatchNorm3d(64*2)
#             self.fcc = 256
#             self.fccbn = nn.BatchNorm3d(256)
#             self.point2 = nn.Conv3d(in_channels=256, out_channels=out_channel, kernel_size=1, stride=1)
#         elif in_channel==128+256 and out_channel==128:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=128, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(128)
#             self.slicon = nn.Conv3d(in_channels=32*2,out_channels=32*2,kernel_size=3,stride=1,padding=1)
#             self.bn2 = nn.BatchNorm3d(32*2)
#             self.fcc = 128
#             self.fccbn = nn.BatchNorm3d(128)
#             self.point2 = nn.Conv3d(in_channels=128, out_channels=out_channel, kernel_size=1, stride=1)
#         elif in_channel==64+128 and out_channel==64:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=64, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(64)
#             self.slicon = nn.Conv3d(in_channels=16*2, out_channels=16*2, kernel_size=3, stride=1, padding=1)
#             self.bn2 = nn.BatchNorm3d(16*2)
#             self.fcc = 64
#             self.fccbn = nn.BatchNorm3d(64)
#             self.point2 = nn.Conv3d(in_channels=64, out_channels=out_channel, kernel_size=1, stride=1)
#         elif in_channel==32+64 and out_channel==32:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=32, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(32)
#             self.slicon = nn.Conv3d(in_channels=8*2, out_channels=8*2, kernel_size=3, stride=1, padding=1)
#             self.bn2 = nn.BatchNorm3d(8*2)
#             self.fcc = 32
#             self.fccbn = nn.BatchNorm3d(32)
#             self.point2 = nn.Conv3d(in_channels=32, out_channels=out_channel, kernel_size=1, stride=1)
#         else:
#             self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1)
#             self.bn1 = nn.BatchNorm3d(in_channel)
#             self.slicon = nn.Conv3d(in_channels=in_channel // 2, out_channels=in_channel // 2, kernel_size=3, stride=1,
#                                     padding=1)
#             self.bn2 = nn.BatchNorm3d(in_channel // 2)
#             self.fcc = 16
#             self.fccbn = nn.BatchNorm3d(16)
#             self.point2 = nn.Conv3d(in_channels=in_channel , out_channels=out_channel, kernel_size=1, stride=1)
#         self.ave_pooling = nn.AdaptiveAvgPool3d(1)
#         self.fc1 = nn.Linear(self.fcc, self.fcc)  # .cuda()
#         self.fc2 = nn.Linear(self.fcc, self.fcc)  # .cuda()
#         self.softmax = f.softmax
#
#         self.ave_pooling = nn.AdaptiveAvgPool3d(1)  # .cuda()   #  全局平均池化
#         self.ave_pooling1 = nn.AdaptiveAvgPool1d(1)  # .cuda()   #  全局平均池化
#         self.fc10 = nn.Linear(4, 2)  # .cuda()
#         self.fc20 = nn.Linear(2, 4)  # .cuda()
#         self.fc0 = nn.Linear(2, 2)  # .cuda()
#     def forward(self,x):
#         # print(x.shape)   #torch.Size([4, 1, 32, 64, 64])
#         x = self.point1(x)
#         x = self.bn1(x)
#         x = self.prelu(x)
#         # print(x.shape)   #torch.Size([4, 16, 32, 64, 64])
#         w = x.shape[1]//2#4
#         # print(w)     #4
#         x = torch.split(x,w,dim=1)
#         # for i in x:
#         #     print(i.size())
#         # print(len(x))
#         x0 = x[0]
#         # x0 = self.slicon(x0)
#         # x0 = self.bn2(x0)
#         # out0 = self.prelu(x0)
#         # x0 = x0
#
#         # x1 = x[1]+x[0]
#         x1 = x[1]
#         x1 = self.slicon(x1)
#         x1 = self.bn2(x1)
#         x1 = self.prelu(x1)
#         # x2 = x[1]+x[2]
#         # x2 = x[2]
#         # x2 = self.slicon(x2)
#         # x2 = self.bn2(x2)
#         # x2 = self.prelu(x2)
#         # x3 = x[2]+x[3]
#         # x3 = x[3]
#         # x3 = self.slicon(x3)
#         # x3 = self.bn2(x3)
#         # x3 = self.prelu(x3)
#         # x = torch.cat((x0,x1,x2,x3),1)
#         x00 = self.ave_pooling(x0)
#         x00 = torch.flatten(x00, 1)  # torch.Size([3, 1])
#         x00 = x00.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
#         x00 = self.ave_pooling1(x00)  # 在添加的维度上对所有channel取平均
#
#         x11 = self.ave_pooling(x1)  # x11工具人
#         x11 = torch.flatten(x11, 1)  # torch.Size([3, 1])
#         x11 = x11.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
#         x11 = self.ave_pooling1(x11)  # 在添加的维度上对所有channel取平均
#
#         out1 = torch.cat((x00,x11), 1)  # 在添加的维度上进行concat
#         out1 = torch.flatten(out1, 1)
#         out1 = self.fc0(out1)
#         out1 = self.prelu(out1)
#         out1 = self.fc0(out1)
#         # out1 = self.sigmoid(out1)
#         # out1 = self.prelu(out1)
#         out1 = self.softmax(out1, 1)
#         out1 = out1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 恢复  #torch.Size([channel,1,1,1])
#         # print(out1[:,0,:].unsqueeze(1).shape)
#         # print(x1.shape)
#         x111 = torch.mul(out1[:, 0, :].unsqueeze(1), x0)
#         x222 = torch.mul(out1[:, 1, :].unsqueeze(1), x1)
#
#         x = torch.cat((x111, x222), 1)  # 尺度注意over
#         # x = torch.cat((x0,x1),1)
#         # # # print("x",x.shape)
#         # xfc0 = self.ave_pooling(x)
#         # xfc0 = torch.flatten(xfc0, start_dim=1)
#         # # # print("xfc0.shape",xfc0.shape)
#         # xfc1 = self.fc1(xfc0)
#         # xfc1 = self.prelu(xfc1)
#         # xfc2 = self.fc2(xfc1)
#         # # xfc2 = self.prelu(xfc2)
#         # xfc2 = self.softmax(xfc2, dim=1)###########pc
#         # # # print("xfc2.shape",xfc2.shape)
#         # xfc2 = xfc2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         # # # print("xfc2.shape", xfc2.shape)
#         # # x = torch.mul(xfc2, x)
#         # x = torch.mul(xfc2, x)
#         # # out00 = self.ave_pooling(out0)
#         # # out11 = self.ave_pooling(out1)
#         # # out22 = self.ave_pooling(out2)
#         # # out33 = self.ave_pooling(out3)
#         # # out00 = out00.squeeze(-1).squeeze(-1).squeeze(-1)
#         # # out11 = out11.squeeze(-1).squeeze(-1).squeeze(-1)
#         # # out22 = out22.squeeze(-1).squeeze(-1).squeeze(-1)
#         # # out33 = out33.squeeze(-1).squeeze(-1).squeeze(-1)
#         # # # print(out11.shape)#([4, 16])
#         # # out00 = out00.unsqueeze(1)
#         # # out11 = out11.unsqueeze(1)
#         # # out22 = out22.unsqueeze(1)
#         # # out33 = out33.unsqueeze(1)
#         # # # print(out11.shape)#([4, 1, 16])
#         # # outf = torch.cat((out00,out11,out22,out33),1)
#         # # # print(outf.shape)#([4, 3, 16])
#         # # outf = self.ave_pooling1(outf)
#         # # # print(outf.shape)#([4, 3, 1])
#         # # outf = torch.flatten(outf, start_dim=1)
#         # #
#         # # outf = self.fc10(outf)
#         # # outf = self.prelu(outf)
#         # # outf = self.fc20(outf)
#         # # outf = self.softmax(outf,dim=1)
#         # #
#         # #
#         # # # print(outf)
#         # # outf = outf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         # # # print(outf.shape)#([4, 3, 1, 1, 1, 1])
#         # # # print(outf[:,0].shape)#([4, 1, 1, 1, 1])
#         # # # print(outf[:,0])
#         # # # print(out1.shape)
#         # # out0 = torch.mul(outf[:,0], out1)
#         # # out1 = torch.mul(outf[:,1], out2)
#         # # out2 = torch.mul(outf[:,2], out3)
#         # # out3 = torch.mul(outf[:,3], out3)
#         # # # print("out.shape", out1.shape)#([4, 16, 32, 64, 64])
#         # # # out = out0.add(out1)
#         # # # out = out.add(out2)
#         # # # out = out.add(out3)#
#         # # out = torch.cat((out0, out1, out2, out3), 1)
#         # # print(out.shape)
#         # # x = self.fccbn(out)##############22
#         # # x = self.prelu(x)
#         # # print(x.shape)
#         x = self.point2(x)
#         return x
#
# class res_block(nn.Module):  ##nn.Module
#     def __init__(self, i_channel, o_channel, lei):
#         super(res_block, self).__init__()
#         self.in_c = i_channel
#         self.out_c = o_channel
#
#         if self.in_c == 1:
#             # self.conv1 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=5, stride=1, padding=2)
#             self.conv1 = r2_block(in_channel=i_channel, out_channel=o_channel).cuda()
#         elif self.in_c == 80:
#             # self.conv1 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=5, stride=1, padding=2)
#             self.conv1 = r2_block(in_channel=i_channel, out_channel=o_channel).cuda()
#         else:
#
#             # self.conv1 = nn.Conv3d(in_channels=i_channel, out_channels=i_channel, kernel_size=5, stride=1, padding=2)
#             self.conv1 = r2_block(in_channel=i_channel, out_channel=i_channel).cuda()
#         # self.conv2 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=5, stride=1, padding=2)
#         self.conv2 = r2_block(in_channel=i_channel, out_channel=o_channel).cuda()
#
#         self.conv3 = nn.Conv3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2, stride=2).cuda()  ###  卷积下采样
#
#         self.conv4 = nn.ConvTranspose3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2, stride=2).cuda()   ###  反卷积上采样
#
#         self.conv5 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=1, stride=1).cuda()   ###  点卷积
#
#         self.bn = nn.BatchNorm3d(i_channel).cuda()
#         self.bn1 = nn.BatchNorm3d(o_channel).cuda()
#         self.prelu = nn.ELU().cuda()
#         self.lei = lei
#         self.drop = nn.Dropout3d()
#
#     def forward(self,x):
#         if self.lei == "forward1":
#             out = self.forward1(x)
#         elif self.lei == "forward2":
#             out = self.forward2(x)
#         elif self.lei == "forward3":
#             out = self.forward3(x)
#         elif self.lei == "deconv":
#             out = self.deconv(x)
#         elif self.lei == "upconv":
#             out = self.upconv(x)
#         else:
#             out = self.pointconv(x)
#         return out
#
#
#
#
#     def forward1(self, x):
#         x = x.to(torch.float32)
#         res = x   ###   记录下输入时的 x
#         res1 = res_block(self.in_c,self.out_c,"pointconv")
#         res = res1(res)
#         # print(x.shape)           ####记下   torch.Size([1, 1, 192, 160, 160])
#         out = self.conv1(x)
#         # print(out.shape)         ####记下   torch.Size([1, 16, 192, 160, 160])
#         out = self.bn1(out)
#         out = self.drop(out)
#         out = self.prelu(out)
#
#         out = res.add(out)
#
#         return out
#
#     def forward2(self,x ):
#         res = x   ###   记录下输入时的 x
#         res1 = res_block(self.in_c, self.out_c, "pointconv")
#         res = res1(res)
#         out = self.conv1(x)
#         out = self.bn(out)
#         out = self.prelu(out)
#         out = self.conv2(out)
#         out = self.bn1(out)
#         out = self.drop(out)
#         out = self.prelu(out)
#         out = res.add(out)
#
#
#         return out
#
#     def forward3(self, x):
#         res = x   ###   记录下输入时的 x
#         res1 = res_block(self.in_c, self.out_c, "pointconv")
#         res = res1(res)
#         out = self.conv1(x)
#         out = self.bn(out)
#         out = self.prelu(out)
#         out = self.conv1(out)
#         out = self.bn(out)
#         out = self.prelu(out)
#         out = self.conv2(out)
#         out = self.bn1(out)
#         out = self.drop(out)
#         out = self.prelu(out)
#         out = res.add(out)
#
#
#         return out
#
#     def deconv(self,x):
#         out = self.conv3(x)
#         out = self.bn(out)
#         out = self.prelu(out)
#         return out
#
#     def upconv(self,x):
#         out = self.conv4(x)
#         out = self.bn(out)
#         out = self.prelu(out)
#         return out
#
#     def pointconv(self,x):
#         out = self.conv5(x)
#         out = self.bn1(out)
#         out = self.prelu(out)
#         return out

class se_block(nn.Module):    #super(se_block, self).__init__()：继承父类nn.Module的初始化方法。
    def __init__(self,in_channel,out_channel,r=16,L=4):
        super(se_block, self).__init__()
        self.in_channel = in_channel   #声明输入通道数，
        self.out_channel = out_channel  #声明输出通道数。
        self.k_3_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,padding=1)  #声明卷积层，输入通道数是in_channel，输出通道数是out_channel，卷积核大小是3，步长是1，padding是1
        self.bn = nn.BatchNorm3d(out_channel)  #声明BatchNorm层，输入的通道数为out_channel
        self.prelu = nn.PReLU()  #声明PReLU激活层。
        self.sigmoid = nn.Sigmoid()  #声明Sigmoid激活层。
        self.ave_pooling = nn.AdaptiveAvgPool3d(1)    #声明AdaptiveAvgPooling层，池化窗口大小为1。
        d = max(out_channel // r,L)      #声明变量d，是out_channel除以r和L的最大值
        self.fc1 = nn.Linear(out_channel, d)  # .cuda()  #声明线性层，输入的特征数是out_channel，输出的特征数是d。
        self.fc2 = nn.Linear(d, out_channel)  # .cuda()  #声明线性层，输入的特征数是d，输出的特征数是out_channel。
    def forward(self,x):
        out = self.k_3_conv(x)  #通过3维卷积层对输入x进行处理。
        out = self.bn(out)    #对卷积层的输出进行批标准化处理。
        out = self.prelu(out)  #对批标准化的输出进行PReLU激活。
        # print(out.shape)
        out1d = self.ave_pooling(out)  ##  （BD,C,1*1*1）   对批标准化的PReLU激活的输出进行平均池化。
        out1d = torch.flatten(out1d, start_dim=1)  #对平均池化的输出做压扁操作，变为一维。
        # print(out1d.shape)
        out_mid = self.fc1(out1d)   #对压扁后的输出经过线性全连接层处理。
        out_mid = self.prelu(out_mid)  #对线性全连接层的输出进行PReLU激活。
        out_out = self.fc2(out_mid)   #对PReLU激活的输出再次经过线性全连接层处理。
        out_out = self.sigmoid(out_out)  #对线性全连接层的输出进行sigmoid激活。
        out_out = out_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    #对sigmoid激活的输出进行扩维操作
        # print(out_out.shape)
        out = torch.mul(out, out_out) #对卷积层的输出与sigmoid激活的输出进行元素级相乘。
        return out  #最终返回相乘后的结果

class r2_block(nn.Module):   #继承了torch.nn.Module，定义了一个名为r2_block的模块类。
    def __init__(self,in_channel,out_channel,s=4):   #in_channel：输入通道数量。 out_channel：输出通道数量。 s：步长，默认值为4。 self.prelu：使用了nn.ELU()，表示对输入使用一个指数线性单元（ELU）进行非线性变换。
        super(r2_block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.prelu = nn.ELU()
        # if in_channel==1 and out_channel==16:
        #     self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=16, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm3d(16)
        #     self.slicon = nn.Conv3d(in_channels=4*2*2,out_channels=4*2*2,kernel_size=3,stride=1,padding=1)
        #     self.bn2 = nn.BatchNorm3d(4*2*2)
        #     self.fcc = 16
        #     self.fccbn = nn.BatchNorm3d(16)
        #     self.point2 = nn.Conv3d(in_channels=16*2, out_channels=out_channel, kernel_size=1, stride=1)
        # elif in_channel==16 and out_channel==32:
        #     self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=32, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm3d(32)
        #     self.slicon = nn.Conv3d(in_channels=8*2*2,out_channels=8*2*2,kernel_size=3,stride=1,padding=1)
        #     self.bn2 = nn.BatchNorm3d(8*2*2)
        #     self.fcc = 32
        #     self.fccbn = nn.BatchNorm3d(32)
        #     self.point2 = nn.Conv3d(in_channels=32*2, out_channels=out_channel, kernel_size=1, stride=1)
        # elif in_channel == 32 and out_channel==64:
        #     self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=64, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm3d(64)
        #     self.slicon = nn.Conv3d(in_channels=16*2*2, out_channels=16*2*2, kernel_size=3, stride=1, padding=1)
        #     self.bn2 = nn.BatchNorm3d(16*2*2)
        #     self.fcc = 64
        #     self.fccbn = nn.BatchNorm3d(64)
        #     self.point2 = nn.Conv3d(in_channels=64*2, out_channels=out_channel, kernel_size=1, stride=1)
        # elif in_channel==64 and out_channel==128:
        #     self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=128, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm3d(128)
        #     self.slicon = nn.Conv3d(in_channels=32*2*2,out_channels=32*2*2,kernel_size=3,stride=1,padding=1)
        #     self.bn2 = nn.BatchNorm3d(32*2*2)
        #     self.fcc = 128
        #     self.fccbn = nn.BatchNorm3d(128)
        #     self.point2 = nn.Conv3d(in_channels=128*2, out_channels=out_channel, kernel_size=1, stride=1)
        # elif in_channel==128 and out_channel==256:
        #     self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=256, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm3d(256)
        #     self.slicon = nn.Conv3d(in_channels=64*2*2,out_channels=64*2*2,kernel_size=3,stride=1,padding=1)
        #     self.bn2 = nn.BatchNorm3d(64*2*2)
        #     self.fcc = 256
        #     self.fccbn = nn.BatchNorm3d(256)
        #     self.point2 = nn.Conv3d(in_channels=256*2, out_channels=out_channel, kernel_size=1, stride=1)
        # elif in_channel==256 and out_channel==256:
        #     self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=256, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm3d(256)
        #     self.slicon = nn.Conv3d(in_channels=64*2*2,out_channels=64*2*2,kernel_size=3,stride=1,padding=1)
        #     self.bn2 = nn.BatchNorm3d(64*2*2)
        #     self.fcc = 256
        #     self.fccbn = nn.BatchNorm3d(256)
        #     self.point2 = nn.Conv3d(in_channels=256*2, out_channels=out_channel, kernel_size=1, stride=1)
        # elif in_channel==128+256 and out_channel==128:
        #     self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=128, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm3d(128)
        #     self.slicon = nn.Conv3d(in_channels=32*2*2,out_channels=32*2*2,kernel_size=3,stride=1,padding=1)
        #     self.bn2 = nn.BatchNorm3d(32*2*2)
        #     self.fcc = 128
        #     self.fccbn = nn.BatchNorm3d(128)
        #     self.point2 = nn.Conv3d(in_channels=128*2, out_channels=out_channel, kernel_size=1, stride=1)
        # elif in_channel==64+128 and out_channel==64:
        #     self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=64, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm3d(64)
        #     self.slicon = nn.Conv3d(in_channels=16*2*2, out_channels=16*2*2, kernel_size=3, stride=1, padding=1)
        #     self.bn2 = nn.BatchNorm3d(16*2*2)
        #     self.fcc = 64
        #     self.fccbn = nn.BatchNorm3d(64)
        #     self.point2 = nn.Conv3d(in_channels=64*2, out_channels=out_channel, kernel_size=1, stride=1)
        # elif in_channel==32+64 and out_channel==32:
        #     self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=32, kernel_size=1, stride=1)
        #     self.bn1 = nn.BatchNorm3d(32)
        #     self.slicon = nn.Conv3d(in_channels=8*2*2, out_channels=8*2*2, kernel_size=3, stride=1, padding=1)
        #     self.bn2 = nn.BatchNorm3d(8*2*2)
        #     self.fcc = 32
        #     self.fccbn = nn.BatchNorm3d(32)
        #     self.point2 = nn.Conv3d(in_channels=32*2*2, out_channels=out_channel, kernel_size=1, stride=1)
        # else:
        #这里定义了4个卷积层：
        # self.point1：一个3D卷积层，输入通道数为in_channel，输出通道数为(in_channel + out_channel) // 2，卷积核大小为1，步长为1。
        # self.bn1：一个批量归一化层，归一化的维度为(in_channel + out_channel) // 2。self.slicon
        self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=(in_channel+out_channel)//2, kernel_size=1, stride=1)  #"//2"表示在代码中使用整除，即向下取整。它表示输入通道数和输出通道数的平均数，即：(in_channel + out_channel)//2。例如，如果in_channel为10，out_channel为20，那么(in_channel + out_channel)//2的值将是15。
        self.bn1 = nn.BatchNorm3d((in_channel+out_channel)//2) #定义了一个三维批标准化层，即BatchNorm3d层(in_channel+out_channel)//2 表示输入通道数，即上一层输出的通道数
        self.slicon = nn.Conv3d(in_channels=(in_channel+out_channel)//2, out_channels=(in_channel+out_channel)//2, kernel_size=3, stride=1,padding=1)  #定义了一个3维卷积层，它接受具有(in_channel+out_channel)//2个通道的输入张量，并产生具有(in_channel+out_channel)//2个通道的输出张量。卷积核的大小是3，步长为1，边缘补零使得输入和输出大小相同。
        self.bn2 = nn.BatchNorm3d((in_channel+out_channel)//2) #定义了一个三维批标准化层，即BatchNorm3d层(in_channel+out_channel)//2 ，用于对上一层的输出进行归一化，以防止激活值的梯度消失。

        # self.fcc = 16
        # self.fccbn = nn.BatchNorm3d(16)
        self.point2 = nn.Conv3d(in_channels=(in_channel+out_channel)//2*2, out_channels=out_channel, kernel_size=1, stride=1)
        #三维卷积操作，它的输入通道数是 (in_channel + out_channel)//2 * 2，输出通道数是 out_channel，卷积核的尺寸为 1 x 1 x 1，步长为 1。
        # 具体来说，这个卷积操作是对输入特征图进行一个降维操作，从 (in_channel + out_channel)//2 * 2 通道变为 out_channel 通道。
        # self.ave_pooling = nn.AdaptiveAvgPool3d(1)
        # self.fc1 = nn.Linear(self.fcc, self.fcc)  # .cuda()
        # self.fc2 = nn.Linear(self.fcc, self.fcc)  # .cuda()
        self.softmax = f.softmax
#创建了两个自适应平均池化层。其中nn.AdaptiveAvgPool3d是一种3维自适应平均池化方法，它可以根据输入的数据维度自动确定输出的维度。参数1指的是输出的长度为1。同样的，nn.AdaptiveAvgPool1d是一种1维自适应平均池化方法。
        self.ave_pooling = nn.AdaptiveAvgPool3d(1)  # .cuda()   #  全局平均池化
        self.ave_pooling1 = nn.AdaptiveAvgPool1d(1)  # .cuda()   #  全局平均池化
        # self.fc10 = nn.Linear(4, 2)  # .cuda()
        # self.fc20 = nn.Linear(2, 4)  # .cuda()
        self.fc0 = nn.Linear(2, 2)  # .cuda()  #创建了一个全连接层，也称为线性层。它将输入数据与全连接层的权重矩阵进行矩阵乘法，再加上偏置项，得到输出数据。这一层的参数是两个：输入的维数为2，输出的维数为2，即该全连接层有两个输入神经元，两个输出神经元。
    def forward(self,x):
        # with torch.no_grad():
        x = self.point1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        # print(x.shape)   #torch.Size([4, 16, 32, 64, 64])
        # w = x.shape[1]//2#4
        # # print(w)     #4
        # x = torch.split(x,w,dim=1)
        #
        # x0 = x[0]
        #
        # x1 = x[1]
        x1 = x
        x1 = self.slicon(x1)
        x1 = self.bn2(x1)
        x1 = self.prelu(x1)
        # x = x1
        #     # x2 = x[1]+x[2]
        #     # x2 = x[2]
        #     # x2 = self.slicon(x2)
        #     # x2 = self.bn2(x2)
        #     # x2 = self.prelu(x2)
        #     # x3 = x[2]+x[3]
        #     # x3 = x[3]
        #     # x3 = self.slicon(x3)
        #     # x3 = self.bn2(x3)
        #     # x3 = self.prelu(x3)
        #     # x = torch.cat((x0,x1,x2,x3),1)
        # # x00 = self.ave_pooling(x0)
        # # x00 = torch.flatten(x00, 1)  # torch.Size([3, 1])
        # # x00 = x00.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
        # # x00 = self.ave_pooling1(x00)  # 在添加的维度上对所有channel取平均
        # #
        # # x11 = self.ave_pooling(x1)  # x11工具人
        # # x11 = torch.flatten(x11, 1)  # torch.Size([3, 1])
        # # x11 = x11.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
        # # x11 = self.ave_pooling1(x11)  # 在添加的维度上对所有channel取平均
        # #
        # # out1 = torch.cat((x00,x11), 1)  # 在添加的维度上进行concat
        # # out1 = torch.flatten(out1, 1)
        # # out1 = self.fc0(out1)
        # # out1 = self.prelu(out1)
        # # out1 = self.fc0(out1)
        # # # out1 = self.sigmoid(out1)
        # # # out1 = self.prelu(out1)
        # # out1 = self.softmax(out1, 1)
        # # out1 = out1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 恢复  #torch.Size([channel,1,1,1])
        # # # print(out1[:,0,:].unsqueeze(1).shape)
        # # # print(x1.shape)
        # # x0 = torch.mul(out1[:, 0, :].unsqueeze(1), x0)
        # # x1 = torch.mul(out1[:, 1, :].unsqueeze(1), x1)
        #
        # x = torch.cat((x0, x1), 1)  # 尺度注意over
        # # x = torch.cat((x0,x1),1)
        # # # # print("x",x.shape)
        # # xfc0 = self.ave_pooling(x)
        # # xfc0 = torch.flatten(xfc0, start_dim=1)
        # # # # print("xfc0.shape",xfc0.shape)
        # # xfc1 = self.fc1(xfc0)
        # # xfc1 = self.prelu(xfc1)
        # # xfc2 = self.fc2(xfc1)
        # # # xfc2 = self.prelu(xfc2)
        # # xfc2 = self.softmax(xfc2, dim=1)###########pc
        # # # # print("xfc2.shape",xfc2.shape)
        # # xfc2 = xfc2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # # # # print("xfc2.shape", xfc2.shape)
        # # # x = torch.mul(xfc2, x)
        # # x = torch.mul(xfc2, x)
        # # out00 = self.ave_pooling(out0)
        # # out11 = self.ave_pooling(out1)
        # # out22 = self.ave_pooling(out2)
        # # out33 = self.ave_pooling(out3)
        # # out00 = out00.squeeze(-1).squeeze(-1).squeeze(-1)
        # # out11 = out11.squeeze(-1).squeeze(-1).squeeze(-1)
        # # out22 = out22.squeeze(-1).squeeze(-1).squeeze(-1)
        # # out33 = out33.squeeze(-1).squeeze(-1).squeeze(-1)
        # # # print(out11.shape)#([4, 16])
        # # out00 = out00.unsqueeze(1)
        # # out11 = out11.unsqueeze(1)
        # # out22 = out22.unsqueeze(1)
        # # out33 = out33.unsqueeze(1)
        # # # print(out11.shape)#([4, 1, 16])
        # # outf = torch.cat((out00,out11,out22,out33),1)
        # # # print(outf.shape)#([4, 3, 16])
        # # outf = self.ave_pooling1(outf)
        # # # print(outf.shape)#([4, 3, 1])
        # # outf = torch.flatten(outf, start_dim=1)
        # #
        # # outf = self.fc10(outf)
        # # outf = self.prelu(outf)
        # # outf = self.fc20(outf)
        # # outf = self.softmax(outf,dim=1)
        # #
        # #
        # # # print(outf)
        # # outf = outf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # # # print(outf.shape)#([4, 3, 1, 1, 1, 1])
        # # # print(outf[:,0].shape)#([4, 1, 1, 1, 1])
        # # # print(outf[:,0])
        # # # print(out1.shape)
        # # out0 = torch.mul(outf[:,0], out1)
        # # out1 = torch.mul(outf[:,1], out2)
        # # out2 = torch.mul(outf[:,2], out3)
        # # out3 = torch.mul(outf[:,3], out3)
        # # # print("out.shape", out1.shape)#([4, 16, 32, 64, 64])
        # # # out = out0.add(out1)
        # # # out = out.add(out2)
        # # # out = out.add(out3)#
        # # out = torch.cat((out0, out1, out2, out3), 1)
        # # print(out.shape)
        # # x = self.fccbn(out)##############22
        # # x = self.prelu(x)
        # # print(x.shape)
        x = torch.cat((x, x1), 1)
        x = self.point2(x)
        return x
    #首先，x通过self.point1这一层卷积，并使用BatchNorm层self.bn1进行归一化处理，再通过PReLU激活函数self.prelu进行激活处理。
    #接下来，计算一个新的张量x1，x1先经过卷积层self.slicon，再经过BN层self.bn2，最后经过PReLU激活函数self.prelu。
    #将x和x1按照维度1（即通道维度）进行拼接，作为最终的x。
    #最后，x通过self.point2这一卷积层，并作为函数的输出返回。

class res_block(nn.Module):  ##nn.Module 继承自PyTorch中的nn.Module的类，名为res_block。该类实现了一个ResNet的残差块。
    def __init__(self, i_channel, o_channel, lei):
        super(res_block, self).__init__()
        self.in_c = i_channel   #输入的通道
        self.out_c = o_channel  #输出通道

        if self.in_c == 1:
            # self.conv1 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=3, stride=1, padding=1)
            self.conv1 = r2_block(in_channel=i_channel, out_channel=o_channel).cuda()      #层使用了一个r2_block的卷积块，它的
        # elif self.in_c == 80:
            # self.conv1 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=5, stride=1, padding=2)
            # self.conv1 = r2_block(in_channel=i_channel, out_channel=o_channel).cuda()
        else:
            # self.conv1 = nn.Conv3d(in_channels=i_channel, out_channels=i_channel, kernel_size=3, stride=1, padding=1)
            self.conv1 = r2_block(in_channel=i_channel, out_channel=i_channel).cuda()
        # self.conv2 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = r2_block(in_channel=i_channel, out_channel=o_channel).cuda()#同上，该层也是一个使用了r2_block的卷积块。

        self.conv3 = nn.Conv3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2, stride=2).cuda()  ###  卷积下采样 该层是一个普通的卷积层，使用了降采样操作。

        self.conv4 = nn.ConvTranspose3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2, stride=2).cuda()   ###  反卷积上采样 该层是一个反卷积层，使用了上采样操作。

        self.conv5 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=1, stride=1).cuda()   ###  点卷积

        self.bn = nn.BatchNorm3d(i_channel).cuda()  #self.bn、self.bn1：分别是对应的batch normalization层。
        self.bn1 = nn.BatchNorm3d(o_channel).cuda()
        self.prelu = nn.ELU().cuda() #该层是一个PReLU激活函数层
        self.lei = lei  #该变量存储了额外的一些参数，其具体作用未知。
        self.drop = nn.Dropout3d()  #该层是一个dropout层。Dropout 是一种正则化方法，用于防止神经网络的过拟合。在训练时，dropout 层会随机丢弃一些神经元（输出为0），从而使得神经网络不能依靠某一个特定的特征对输入进行预测，从而强制网络学习更多的特征。在测试时，dropout 层

    def forward(self,x):
        if self.lei == "forward1":
            out = self.forward1(x)
        elif self.lei == "forward2":
            out = self.forward2(x)
        elif self.lei == "forward3":
            out = self.forward3(x)
        elif self.lei == "deconv":
            out = self.deconv(x)
        elif self.lei == "upconv":
            out = self.upconv(x)
        else:
            out = self.pointconv(x)
        return out




    def forward1(self, x):
        x = x.to(torch.float32)
        res = x   ###   记录下输入时的 x
        res1 = res_block(self.in_c,self.out_c,"pointconv")
        res = res1(res)
        # print(x.shape)           ####记下   torch.Size([1, 1, 192, 160, 160])
        out = self.conv2(x)
        # print(out.shape)         ####记下   torch.Size([1, 16, 192, 160, 160])
        out = self.bn1(out)
        out = self.drop(out)
        out = self.prelu(out)

        out = res.add(out)

        return out

    def forward2(self,x ):
        res = x   ###   记录下输入时的 x
        res1 = res_block(self.in_c, self.out_c, "pointconv")
        res = res1(res)
        out = self.conv1(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.drop(out)
        out = self.prelu(out)
        out = res.add(out)


        return out

    def forward3(self, x):
        res = x   ###   记录下输入时的 x
        res1 = res_block(self.in_c, self.out_c, "pointconv")
        res = res1(res)
        out = self.conv1(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv1(out)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.drop(out)
        out = self.prelu(out)
        out = res.add(out)


        return out

    def deconv(self,x):
        out = self.conv3(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

    def upconv(self,x):
        out = self.conv4(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

    def pointconv(self,x):
        out = self.conv5(x)
        out = self.bn1(out)
        out = self.prelu(out)
        return out