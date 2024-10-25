import torch
import torch.nn as nn
import torch.nn.functional as F
import math

################################## EfficientAttention
class EfficientAttention(nn.Module):
    def __init__(self, in_channels, heads=8, dropout=0.1):
        super(EfficientAttention, self).__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.dim_per_head = in_channels // heads
        self.query_linear = nn.Linear(in_channels, in_channels)
        self.key_linear = nn.Linear(in_channels, in_channels)
        self.value_linear = nn.Linear(in_channels, in_channels)
        self.fc_out = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, C, T, V = x.size()
        
        # 先将输入转化为 (N * V, T, C) 形状
        x = x.permute(0, 3, 2, 1).contiguous().view(-1, T, C)

        queries = self.query_linear(x).view(N * V, T, self.heads, self.dim_per_head).transpose(1, 2)  # (N * V, heads, T, dim_per_head)
        keys = self.key_linear(x).view(N * V, T, self.heads, self.dim_per_head).transpose(1, 2)
        values = self.value_linear(x).view(N * V, T, self.heads, self.dim_per_head).transpose(1, 2)

        attention_scores = torch.einsum("nhid,nhjd->nhij", queries, keys) / (self.dim_per_head ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        out = torch.einsum("nhij,nhjd->nhid", attention_weights, values).reshape(N * V, T, C)
        out = self.fc_out(out)

        # 转回 (N, V, T, C)
        out = out.view(N, V, T, C).permute(0, 3, 2, 1)

        return out
        
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
        # x shape: (N, C, T*V)
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
    def __init__(self, num_classes, in_channels, local_size=5):
        super(ActionRecognitionModel, self).__init__()
        self.mlca = MLCA(in_channels, local_size)

    def forward(self, x):
        # x shape: (N, C, T, V)
        # print("Input shape:", x.shape)  # 打印输入形状

        N, C, T, V = x.shape
        # x = x.view(N, C, -1)  # (N, C, T*V)
        x = x.reshape(N, C, -1)  # (N, C, T*V)


        # print("Shape after view:", x.shape)  # 打印变换后的形状
        
        x = self.mlca(x)  # Apply MLCA
        
        # 确保 MLCA 的输出形状符合要求
        # print("Shape after MLCA:", x.shape)  # 检查 MLCA 输出形状

        return x  # 返回 (N, C, T, V)


