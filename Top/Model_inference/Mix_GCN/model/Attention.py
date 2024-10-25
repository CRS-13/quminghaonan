import torch
import torch.nn as nn
import torch.nn.functional as F
import math

######################################### SEAttention
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze and Excitation
        se = self.avg_pool(x).view(b, c)
        se = self.fc(se).view(b, c, 1, 1)
        return x * se.expand_as(x)  # 逐通道相乘


######################################### MLCA
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.local_weight = local_weight
        
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, t_v = x.shape
        V = 17  # 根据具体数据集设定 V 的值
        T = t_v // V  # 计算 T
        
        x = x.view(b, c, T, V)  # 重塑为 (N, C, T, V)

        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)

        # 其他处理不变，确保输出形状为 (N, C, T, V)
        
        # 最终输出保持 (N, C, T, V)
        return x  # 直接返回


class ActionRecognitionModel(nn.Module):
    def __init__(self, in_channels, local_size=5):
        super(ActionRecognitionModel, self).__init__()
        self.mlca = MLCA(in_channels, local_size)
        self.se_attention = SEAttention(channel=in_channels)  # 添加 SEAttention

    def forward(self, x):
        # print(x.shape)
        N, C, T, V = x.shape
        x = x.reshape(N, C, -1)  # (N, C, T*V)

        # x = self.mlca(x)  # Apply MLCA

        # 在此应用 SEAttention
        x = x.view(N, C, T, V)  # 重新调整为 (N, C, T, V) 形状
        x = self.se_attention(x)  # 应用 SEAttention

        return x  # 返回 (N, C, T, V)


def main():
    # 设置参数
    N = 4  # 批量大小
    C = 64  # 通道数
    T = 300  # 时间步
    V = 17  # 关键点数
    num_classes = 10  # 类别数

    # 创建输入张量
    input_tensor = torch.randn(N, C, T, V)

    # 实例化动作识别模型
    model = ActionRecognitionModel(num_classes=num_classes, in_channels=C)

    # 进行前向传播
    output = model(input_tensor)

    # 输出结果的形状
    print("Output shape:", output.shape)  # 应该是 (N, C, T, V)

if __name__ == "__main__":
    main()
