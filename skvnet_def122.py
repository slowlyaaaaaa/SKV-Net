import torch.nn as nn
import torch
import torch.nn.functional as f


class r2_block(nn.Module):  # 继承了torch.nn.Module，定义了一个名为r2_block的模块类。
    def __init__(self, in_channel, out_channel, s=4):  # in_channel：输入通道数量。 out_channel：输出通道数量。 s：步长，默认值为4。 self.prelu：使用了nn.ELU()，表示对输入使用一个指数线性单元（ELU）进行非线性变换。
        super(r2_block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.prelu = nn.ELU()  # 这里使用的是 ELU (Exponential Linear Unit) 激活函数。该函数可以在激活函数的传统范围内产生负值的输出，这有助于模型在训练时更好地捕获数据中的非线性关系
        self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=(in_channel + out_channel) // 2, kernel_size=1, stride=1)# "//2"表示在代码中使用整除，即向下取整。它表示输入通道数和输出通道数的平均数，即：(in_channel + out_channel)//2。例如，如果in_channel为10，out_channel为20，那么(in_channel + out_channel)//2的值将是15。
        self.bn1 = nn.BatchNorm3d( (in_channel + out_channel) // 2)  # 定义了一个三维批标准化层，即BatchNorm3d层(in_channel+out_channel)//2 表示输入通道数，即上一层输出的通道数
        self.slicon = nn.Conv3d(in_channels=(in_channel + out_channel) // 2, out_channels=(in_channel + out_channel) // 2, kernel_size=3, stride=1, padding=1)
        # 定义了一个3维卷积层，它接受具有(in_channel+out_channel)//2个通道的输入张量，并产生具有(in_channel+out_channel)//2个通道的输出张量。卷积核的大小是3，步长为1，边缘补零使得输入和输出大小相同。
        self.bn2 = nn.BatchNorm3d((in_channel + out_channel) // 2)  # 定义了一个三维批标准化层，即BatchNorm3d层(in_channel+out_channel)//2 ，用于对上一层的输出进行归一化，以防止激活值的梯度消失。
        self.point2 = nn.Conv3d(in_channels=(in_channel + out_channel) // 2 * 2, out_channels=out_channel, kernel_size=1, stride=1)
        # self.softmax = f.softmax
        # 创建了两个自适应平均池化层。其中nn.AdaptiveAvgPool3d是一种3维自适应平均池化方法，它可以根据输入的数据维度自动确定输出的维度。参数1指的是输出的长度为1。同样的，nn.AdaptiveAvgPool1d是一种1维自适应平均池化方法。
        # self.ave_pooling = nn.AdaptiveAvgPool3d(1)  # .cuda()   #  全局平均池化
        # self.ave_pooling1 = nn.AdaptiveAvgPool1d(1)  # .cuda()   #  全局平均池化

        self.fc0 = nn.Linear(2,2)  # .cuda()  #创建了一个全连接层，也称为线性层。它将输入数据与全连接层的权重矩阵进行矩阵乘法，再加上偏置项，得到输出数据。这一层的参数是两个：输入的维数为2，输出的维数为2，即该全连接层有两个输入神经元，两个输出神经元。

    def forward(self, x):
        # with torch.no_grad():
        x = self.point1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x1 = x
        x1 = self.slicon(x1)
        x1 = self.bn2(x1)
        x1 = self.prelu(x1)

        x = torch.cat((x, x1), 1)
        x = self.point2(x)
        return x
    # 首先，x通过self.point1这一层卷积，并使用BatchNorm层self.bn1进行归一化处理，再通过PReLU激活函数self.prelu进行激活处理。
    # 接下来，计算一个新的张量x1，x1先经过卷积层self.slicon，再经过BN层self.bn2，最后经过PReLU激活函数self.prelu。
    # 将x和x1按照维度1（即通道维度）进行拼接，作为最终的x。
    # 最后，x通过self.point2这一卷积层，并作为函数的输出返回。


class res_block(nn.Module):  ##nn.Module 继承自PyTorch中的nn.Module的类，名为res_block。该类实现了一个ResNet的残差块。
    def __init__(self, i_channel, o_channel, lei):
        super(res_block, self).__init__()
        self.in_c = i_channel  # 输入的通道
        self.out_c = o_channel  # 输出通道

        if self.in_c == 1:  # 判断输入通道数是否为 1
            self.conv1 = r2_block(in_channel=i_channel,
                                  out_channel=o_channel).cuda()  # 层使用了一个r2_block的卷积块，它的如果输入通道数为 1，则使用一个名为 r2_block的卷积块，否则使用一个输入通道数等于输出通道数的 r2_block 卷积块。这里使用了CUDA 计算。
        else:
            self.conv1 = r2_block(in_channel=i_channel,
                                  out_channel=i_channel).cuda()  # 计算得到的输出通道数通过out_channel=i_channel最后得到输出通道和输入一样的大小，而不是先得到输出在转换
        self.conv2 = r2_block(in_channel=i_channel, out_channel=o_channel).cuda()
        # 同上，该层也是一个使用了r2_block的卷积块。 与第一个卷积块相同。in_channel 是上一层卷积层的输出通道数，也就是输入的特征图的通道数。而输出通道 out_channel 是这个 r2_block 模块的输出通道数，由模型设计者指定。在这里，输出通道数为 o_channel。

        self.conv3 = nn.Conv3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2,
                               stride=2).cuda()  ###  卷积下采样 该层是一个普通的卷积层，使用了降采样操作。
        # 在这一层中，输入通道和输出通道都来自于上一层的输出通道数，也就是前一层的输出特征图的通道数
        self.conv4 = nn.ConvTranspose3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2,
                                        stride=2).cuda()  ###  反卷积上采样 该层是一个反卷积层，使用了上采样操作。
        self.conv5 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=1, stride=1).cuda()  ###  点卷积
        self.bn = nn.BatchNorm3d(i_channel).cuda()  # self.bn、self.bn1：分别是对应的batch normalization层。
        self.bn1 = nn.BatchNorm3d(o_channel).cuda()
        self.prelu = nn.ELU().cuda()  # 该层是一个PReLU激活函数层
        self.lei = lei  # 该变量存储了额外的一些参数，其具体作用未知。
        self.drop = nn.Dropout3d()  # 该层是一个dropout层。Dropout 是一种正则化方法，用于防止神经网络的过拟合。在训练时，dropout 层会随机丢弃一些神经元（输出为0），从而使得神经网络不能依靠某一个特定的特征对输入进行预测，从而强制网络学习更多的特征。在测试时，dropout 层

    def forward(self, x):
        if self.lei == "forward1":
            out = self.forward1(x)
        elif self.lei == "forward2":
            out = self.forward2(x)
        elif self.lei == "forward3":
            out = self.forward3(x)
        elif self.lei == "deconv":  # 反卷积
            out = self.deconv(x)
        elif self.lei == "upconv":
            out = self.upconv(x)  # 上采样
        else:
            out = self.pointconv(x)  # 点卷积
        return out

    def forward1(self, x):
        x = x.to(torch.float32)
        res = x  ###   记录下输入时的 x
        res1 = res_block(self.in_c, self.out_c, "pointconv")
        res = res1(res)
        # print(x.shape)           ####记下   torch.Size([1, 1, 192, 160, 160])
        out = self.conv2(x)
        # print(out.shape)         ####记下   torch.Size([1, 16, 192, 160, 160])
        out = self.bn1(out)
        out = self.drop(out)
        out = self.prelu(out)
        out = res.add(out)  # res和add相加得到最后输出结果过
        return out

    def forward2(self, x):
        res = x  ###   记录下输入时的 x
        res1 = res_block(self.in_c, self.out_c, "pointconv")  # 点卷积，链接
        res = res1(res)  # 后续残差
        out = self.conv1(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.drop(out)
        out = self.prelu(out)
        out = res.add(out)  # res和add相加得到最后输出结果过残差过程
        return out

    def forward3(self, x):
        res = x  ###   记录下输入时的 x
        res1 = res_block(self.in_c, self.out_c, "pointconv")  # 点卷积，链接
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
        out = res.add(out)  # res和add相加得到最后输出结果过
        return out

    def deconv(self, x):  # 反卷积
        out = self.conv3(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

    def upconv(self, x):  # 上采样
        out = self.conv4(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

    def pointconv(self, x):
        out = self.conv5(x)
        out = self.bn1(out)
        out = self.prelu(out)
        return out

    # ┌─────────────┐
    # │ 1x1 conv3d  │
    # └─────────────┘
    #        │
    #        ▼
    # ┌─────────────┐
    # │  BatchNorm3d │
    # └─────────────┘
    #        │
    #        ▼
    # ┌─────────────┐
    # │     ELU     │
    # └─────────────┘
    #        │
    #        ▼
    # ┌─────────────┐
    # │ 3x3 conv3d  │
    # └─────────────┘
    #        │
    #        ▼
    # ┌─────────────┐
    # │  BatchNorm3d │
    # └─────────────┘
    #        │
    #        ▼
    # ┌─────────────┐
    # │     ELU     │
    # └─────────────┘
    #        │
    #        ▼
    # ┌─────────────┐
    # │ 1x1 conv3d  │
    # └─────────────┘
    #        │
    #        ▼
    #      Concat
    #        │
    #        ▼
    # ┌─────────────┐
    # │ 1x1 conv3d  │
    # └─────────────┘
    #        │
    #        ▼
    #      Output
