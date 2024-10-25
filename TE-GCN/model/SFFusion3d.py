import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeAndExcitation2D(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation2D, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            activation,
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (N, C, T, V)
        b, c, t, v = x.size()
        # 使用平均池化以获得每个通道在时间和关节点上的特征
        weighting = F.adaptive_avg_pool2d(x.reshape(b * c, t, v), (1, 1)).reshape(b, c)  # (N, C)
        weighting = self.fc(weighting).reshape(b, c, 1, 1)  # (N, C, 1, 1)
        y = x * weighting  # 广播相乘
        return y

class SqueezeAndExciteFusionAdd2D(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd2D, self).__init__()
        self.se_1 = SqueezeAndExcitation2D(channels_in, activation=activation)
        self.se_2 = SqueezeAndExcitation2D(channels_in, activation=activation)

    def forward(self, se1, se2):
        se1 = self.se_1(se1)
        se2 = self.se_2(se2)
        out = se1 + se2
        return out

# 示例用法
if __name__ == "__main__":
    # 假设的输入数据
    input_1 = torch.randn(112, 128, 150, 17)  # 输入 N C T V
    input_2 = torch.randn(112, 128, 150, 17)  # 同上

    # 打印输入数据的形状
    print(input_1.size())  # 输出: (112, 128, 150, 17)
    print(input_2.size())  # 输出: (112, 128, 150, 17)

    # 创建SqueezeAndExciteFusionAdd2D模块的实例
    block = SqueezeAndExciteFusionAdd2D(channels_in=128)

    # 将输入通过SqueezeAndExciteFusionAdd2D模块获得输出
    output = block(input_1, input_2)

    # 打印输出数据的形状
    print(output.size())  # 输出应该和输入形状相同: (112, 128, 150, 17)
