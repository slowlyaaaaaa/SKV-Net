import torch  # 导入 PyTorch 库。
import torch.nn as nn  # 导入 PyTorch 的神经网络模块。
import skvnet_def122 as vnet_def  # 导入一个名为 skvnet_def122 的自定义模块，并将其重命名为 vnet_def。
import torch.nn.functional as f  # 导入 PyTorch 中的函数模块，并将其重命名为 f。


class SKVNet(nn.Module):  # 定义一个类，名为 SKVNet，继承自 nn.Module 类。
    def __init__(self, num_classes=2):  # 定义初始化函数，该函数的参数 num_classes 默认值为 2。
        super(SKVNet, self).__init__()  # 调用父类的初始化函数。
        self.layer1 = vnet_def.res_block(1, 16,"forward1")  # 从 self.layer1 到 self.layer11111，定义了多个输入数据经过的卷积块，每个卷积块使用了 vnet_def.res_block 函数。
        self.layer11 = vnet_def.res_block(16 + 32, 16, "forward1")
        self.layer111 = vnet_def.res_block(16 + 16 + 32, 16, "forward1")
        self.layer1111 = vnet_def.res_block(16 + 16 + 16 + 32, 16, "forward1")
        self.layer11111 = vnet_def.res_block(16 + 16 + 16 + 16 + 32, 16, "forward1")
        self.layer2 = vnet_def.res_block(16, 32, "forward2")
        self.layer22 = vnet_def.res_block(16 + 32 + 64, 32, "forward2")
        self.layer222 = vnet_def.res_block(16 + 32 + 32 + 64, 32, "forward2")
        self.layer2222 = vnet_def.res_block(16 + 32 + 32 + 32 + 64, 32, "forward2")
        self.layer3 = vnet_def.res_block(32, 64, "forward3")
        self.layer33 = vnet_def.res_block(32 + 64 + 128, 64, "forward3")
        self.layer333 = vnet_def.res_block(32 + 64 + 64 + 128, 64, "forward3")
        self.layer4 = vnet_def.res_block(64, 128, "forward3")
        self.layer44 = vnet_def.res_block(64 + 128 + 256, 128, "forward3")
        self.layer5 = vnet_def.res_block(128, 256, "forward3")
        self.layer1down = vnet_def.res_block(16, 16, "deconv")  # 定义了多个卷积块用于上采样和下采样。
        self.layer2down = vnet_def.res_block(32, 32, "deconv")
        self.layer3down = vnet_def.res_block(64, 64, "deconv")
        self.layer4down = vnet_def.res_block(128, 128, "deconv")
        self.layer5up = vnet_def.res_block(256, 256, "upconv")
        self.layer4up = vnet_def.res_block(128, 128, "upconv")
        self.layer3up = vnet_def.res_block(64, 64, "upconv")
        self.layer2up = vnet_def.res_block(32, 32, "upconv")
        self.layer10 = vnet_def.res_block(16 * 4, num_classes,"pointconv")  ###   num_classes=2    定义一个卷积块，输出的特征数量为 num_classes，该参数的默认值为 2。
        self.softmax = f.softmax  # 定义一个 softmax 函数，用于对输出的概率进行归一化。
        self.prelu = nn.ELU()  # 创建了一个 PyTorch 的 Exponential Linear Unit (ELU) 层并将其赋值给 self.prelu。 ELU 是一种激活函数，常用于深度学习模型的隐藏层，它的输出为输入的指数加上 1，并将负数设置为 0

    # res_block是一个残差模块，在此代码中有20个不同的残差模块。它在输入的特征图上施加一些操作，使得特征图有所改变。这些操作是由一些卷积，激活和归一化层组成的。pointconv是一个全卷积层，它通过一个1x1的卷积核将特征图变成一个点预测。
    ####  提取特征

    def forward(self, x):
        # with torch.no_grad():
        x = self.layer1(x)  # 将输入 x 传入第一个卷积层。
        link1 = x  # 将该层的输出保存到 link1 中，这个变量会在后续的过程中用到。
        x = self.layer1down(x)  # 对该层的输出进行下采样操作。
        x = self.layer2(x)  # 将该层的下采样结果传入第二个卷积层。
        link2 = x  # 将该层的输出保存到 link2 中，这个变量会在后续的过程中用到。
        out1 = self.layer2up(x)  # 将该层的卷积结果传入上采样层，得到一个上采样后的输出。
        out1 = torch.cat((out1, link1), 1)  # 将 link1 和上采样后的输出按通道维度拼接在一起
        out1 = self.layer11(out1)  # 将拼接后的结果传入一个卷积层。
        link5 = out1  # 将该层的输出保存到 link5 中，这个变量会在后续的过程中用到。
        x = self.layer2down(x)  # 对该层的输出进行下采样操作。
        x = self.layer3(x)  # 将该层的下采样结果传入第三个卷积层。
        link3 = x  # 将该层的输出保存到 link3 中，这个变量会在后续的过程中用到。
        out2 = self.layer3up(x)  # 将该层的卷积结果传入上采样层，得到一个上采样后的输出。
        out21 = self.layer1down(link5)  # 对 link5 进行下采样操作。
        out2 = torch.cat((out2, link2, out21), 1)  # 将 link2 和 out21 和上采样后的输出按通道维度拼接在一起。
        out2 = self.layer22(out2)  # 将拼接后的结果传入一个卷积层。
        link6 = out2  # 将该层的输出保存到 link6 中，这个变量会在后续的过程中用到
        out2 = self.layer2up(out2)  # 将该层的卷积结果传入上采样层，得到一个上采样后的输出
        out2 = torch.cat((out2, link1, link5), 1)  # 将out2、link1和link5按照维度1（列）拼接，得到一个新的张量out2，用于后续的计算。
        out2 = self.layer111(out2)  # 对out2进行一层卷积操作，使用了self.layer111定义的卷积层。卷积层可以提取特征，将输入张量变换为一个新的特征张量。
        link8 = out2  # 将经过卷积操作的out2保存到link8中，以便后续的计算使用。
        x = self.layer3down(x)  # 对输入的x进行一层卷积操作，并使用池化操作对张量进行下采样，得到一个下采样后的特征张量。
        x = self.layer4(x)  # 对下采样后的特征张量x进行一层卷积操作，提取特征。
        link4 = x  # 将提取出的特征保存到link4中，以便后续的计算使用。
        out3 = self.layer4up(x)  # 对上一步得到的特征进行上采样操作，并进行一层卷积操作，得到一个新的特征张量out3。
        out31 = self.layer2down(link6)  # 将link6进行一次下采样，并进行一层卷积操作，得到一个新的特征张量out31。
        out3 = torch.cat((out31, out3, link3), 1)  # 将out31、out3、link3按照维度1（列）拼接，得到一个新的张量out3。
        out3 = self.layer33(out3)  # 对新的张量out3进行一次卷积操作，提取特征。
        link7 = out3  # 将经过卷积操作的out3保存到link7中，以便后续的计算使用。
        out3 = self.layer3up(out3)  # 对out3进行一次上采样，并进行一次卷积操作，得到一个新的特征张量out3。
        out32 = self.layer1down(link8)  # 将link8进行一次下采样，并进行一层卷积操作，得到一个新的特征张量out32。
        out3 = torch.cat((out3, out32, link6, link2), 1)  # 将out3、out32、link6、link2按照维度1（列）拼接，得到一个新的张量out3
        out3 = self.layer222(out3)  # 将out3输入到layer222中进行卷积操作，并将输出结果赋值给out3。
        link9 = out3  # 将out3赋值给link9，保存中间结果。
        out3 = self.layer2up(out3)  # 将out3输入到layer2up中进行反卷积操作，并将输出结果赋值给out3。
        out3 = torch.cat((out3, link1, link5, link8), 1)  # 对out3、link1、link5和link8进行拼接，其中1表示在通道维度进行拼接，最终得到拼接后的out3
        out3 = self.layer1111(out3)  # 将out3输入到layer1111中进行卷积操作，并将输出结果赋值给out3。
        link10 = out3  # 将out3赋值给link10，保存中间结果。
        x = self.layer4down(x)  # 将输入x输入到layer4down中进行卷积操作。
        x = self.layer5(x)  # 将上一步的输出结果输入到layer5中进行卷积操作，并将输出结果赋值给x。
        out4 = self.layer5up(x)  # 将x输入到layer5up中进行反卷积操作，并将输出结果赋值给out4。
        out41 = self.layer3down(link7)  # 将link7输入到layer3down中进行卷积操作，并将输出结果赋值给out41。
        out4 = torch.cat((out4, link4, out41), 1)  # 对out4、link4和out41进行拼接，其中1表示在通道维度进行拼接，最终得到拼接后的out4。
        out4 = self.layer44(out4)  # 将out4输入到layer44中进行卷积操作，并将输出结果赋值给out4。
        out4 = self.layer4up(out4)  # 将out4输入到layer4up中进行反卷积操作，并将输出结果赋值给out4。
        out42 = self.layer2down(link9)  # 将link9输入到layer2down中进行卷积操作，并将输出结果赋值给out42。
        out4 = torch.cat((out4, out42, link7, link3), 1)  # 对out4、out42、link7和link3进行拼接，其中1表示在通道维度进行拼接，最终得到拼接后的out4。
        out4 = self.layer333(out4)  # 将out4输入到layer333中进行卷积操作，并将输出结果赋值给out4
        out4 = self.layer3up(out4)  # 上采样out4的分辨率，得到更高分辨率的输出。
        out43 = self.layer1down(link10)  # 下采样link10的分辨率，得到低分辨率的输出，将其与out4进行拼接。
        out4 = torch.cat((out4, out43, link2, link6, link9), 1)  # 对out4、out43、link2、link6、link9进行拼接，得到最终的输出。
        out4 = self.layer2222(out4)  # 应用卷积层对拼接后的输出进行处理。
        out4 = self.layer2up(out4)  # 再次上采样，得到更高分辨率的输出。
        out4 = torch.cat((out4, link1, link5, link8, link10), 1)  # 对out4、link1、link5、link8、link10进行拼接，得到最终的输出。
        out4 = self.layer11111(out4)  # 应用卷积层对拼接后的输出进行处理。

        out = torch.cat((out1, out2, out3, out4), 1)  # 对out1、out2、out3、out4进行拼接，得到最终的输出。

        out = self.layer10(out)  # .pointconv(out)# 应用卷积层对拼接后的输出进行处理。
        out = self.softmax(out, dim=1)  # 应用softmax函数进行归一化，得到每个像素属于不同类别的概率分布。

        return out  # 返回输出。

# class SKVNet(nn.Module):
#     def __init__(self,num_classes=2):
#         super(SKVNet, self).__init__()
#         self.layer1 = vnet_def.res_block(1, 16, "forward1")
#         self.layer11 = vnet_def.res_block(16 + 32, 16, "forward1")
#         self.layer111 = vnet_def.res_block(16 + 16 + 32, 16, "forward1")
#         self.layer1111 = vnet_def.res_block(16 + 16 + 16 + 32, 16, "forward1")
#         self.layer11111 = vnet_def.res_block(16 + 16 + 16 + 16 + 32, 16, "forward1")
#         self.layer2 = vnet_def.res_block(16, 32, "forward2")
#         self.layer22 = vnet_def.res_block(16 + 32 + 64, 32, "forward2")
#         self.layer222 = vnet_def.res_block(16 + 32 + 32 + 64, 32, "forward2")
#         self.layer2222 = vnet_def.res_block(16 + 32 + 32 + 32 + 64, 32, "forward2")
#         self.layer3 = vnet_def.res_block(32, 64, "forward3")
#         self.layer33 = vnet_def.res_block(32 + 64 + 128, 64, "forward3")
#         self.layer333 = vnet_def.res_block(32 + 64 + 64 + 128, 64, "forward3")
#         self.layer4 = vnet_def.res_block(64, 128, "forward3")
#         self.layer44 = vnet_def.res_block(64 + 128 + 256, 128, "forward3")
#         self.layer5 = vnet_def.res_block(128, 256, "forward3")
#         self.layer1down = vnet_def.res_block(16, 16, "deconv")
#         self.layer2down = vnet_def.res_block(32, 32, "deconv")
#         self.layer3down = vnet_def.res_block(64, 64, "deconv")
#         self.layer4down = vnet_def.res_block(128, 128, "deconv")
#         self.layer5up = vnet_def.res_block(256, 256, "upconv")
#         self.layer4up = vnet_def.res_block(128, 128, "upconv")
#         self.layer3up = vnet_def.res_block(64, 64, "upconv")
#         self.layer2up = vnet_def.res_block(32, 32, "upconv")
#
#         self.layer10 = vnet_def.res_block(16 * 4, num_classes, "pointconv")  ###   num_classes=2
#         self.softmax = f.softmax
#         self.ave_pooling = nn.AdaptiveAvgPool3d(1)  # .cuda()   #  全局平均池化
#         self.ave_pooling1 = nn.AdaptiveAvgPool1d(1)  # .cuda()   #  全局平均池化
#         self.fc0 = nn.Linear(4, 4)  # .cuda()
#         self.prelu = nn.ELU()
#
#         ####  提取特征
#
#     def forward(self,x):
#         # with torch.no_grad():
#         x = self.layer1(x)
#         link1 = x
#         x = self.layer1down(x)
#         x = self.layer2(x)
#         link2 = x
#         out1 = self.layer2up(x)
#         out1 = torch.cat((out1,link1),1)
#         out1 = self.layer11(out1)
#         link5 = out1
#         x = self.layer2down(x)
#         x = self.layer3(x)
#         link3 = x
#         out2 = self.layer3up(x)
#         out21 = self.layer1down(link5)
#         out2 = torch.cat((out2,link2,out21),1)
#         out2 = self.layer22(out2)
#         link6 = out2
#         out2 = self.layer2up(out2)
#         out2 = torch.cat((out2,link1,link5),1)
#         out2 = self.layer111(out2)
#         link8 = out2
#         x = self.layer3down(x)
#         x = self.layer4(x)
#         link4 = x
#         out3 = self.layer4up(x)
#         out31 = self.layer2down(link6)
#         out3 = torch.cat((out3,link3,out31),1)
#         out3 = self.layer33(out3)
#         link7 = out3
#         out3 = self.layer3up(out3)
#         out32 = self.layer1down(link8)
#         out3 = torch.cat((out3,link6,link2,out32),1)
#         out3 = self.layer222(out3)
#         link9 = out3
#         out3 = self.layer2up(out3)
#         out3 = torch.cat((out3,link1,link5,link8),1)
#         out3 = self.layer1111(out3)
#         link10 = out3
#         x = self.layer4down(x)
#         x = self.layer5(x)
#         out4 = self.layer5up(x)
#         out41 = self.layer3down(link7)
#         out4 = torch.cat((out4,link4,out41),1)
#         out4 = self.layer44(out4)
#         out4 = self.layer4up(out4)
#         out42 = self.layer2down(link9)
#         out4 = torch.cat((out4,link7,link3,out42),1)
#         out4 = self.layer333(out4)
#         out4 = self.layer3up(out4)
#         out43 = self.layer1down(link10)
#         out4 = torch.cat((out4,link2,link6,link9,out43),1)
#         out4 = self.layer2222(out4)
#         out4 = self.layer2up(out4)
#         out4 = torch.cat((out4,link1,link5,link8,link10),1)
#         out4 = self.layer11111(out4)
#
#         # x00 = self.ave_pooling(out1)
#         # x00 = torch.flatten(x00, 1)  # torch.Size([3, 1])
#         # x00 = x00.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
#         # x00 = self.ave_pooling1(x00)  # 在添加的维度上对所有channel取平均
#         #
#         # x11 = self.ave_pooling(out2)  # x11工具人
#         # x11 = torch.flatten(x11, 1)  # torch.Size([3, 1])
#         # x11 = x11.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
#         # x11 = self.ave_pooling1(x11)  # 在添加的维度上对所有channel取平均
#         #
#         # x22 = self.ave_pooling(out3)  # x11工具人
#         # x22 = torch.flatten(x22, 1)  # torch.Size([3, 1])
#         # x22 = x22.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
#         # x22 = self.ave_pooling1(x22)  # 在添加的维度上对所有channel取平均
#         #
#         # x33 = self.ave_pooling(out4)  # x11工具人
#         # x33 = torch.flatten(x33, 1)  # torch.Size([3, 1])
#         # x33 = x33.unsqueeze(1)  # torch.Size([3, "1", 1])  batchsize,添加的维度,channel
#         # x33 = self.ave_pooling1(x33)  # 在添加的维度上对所有channel取平均
#         #
#         # out1234 = torch.cat((x00, x11,x22,x33), 1)  # 在添加的维度上进行concat
#         # out1234 = torch.flatten(out1234, 1)
#         # out1234 = self.fc0(out1234)
#         # out1234 = self.prelu(out1234)
#         # out1234 = self.fc0(out1234)
#         # out1234 = self.softmax(out1234, 1)
#         # out1234 = out1234.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 恢复  #torch.Size([channel,1,1,1])
#         #
#         # out1 = torch.mul(out1234[:, 0, :].unsqueeze(1), out1)
#         # out2 = torch.mul(out1234[:, 1, :].unsqueeze(1), out2)
#         # out3 = torch.mul(out1234[:, 2, :].unsqueeze(1), out3)
#         # out4 = torch.mul(out1234[:, 3, :].unsqueeze(1), out4)
#
#         out = torch.cat((out1, out2,out3,out4), 1)
#
#         out = self.layer10(out)#.pointconv(out)
#         out = self.softmax(out,dim=1)
#
#         return out
