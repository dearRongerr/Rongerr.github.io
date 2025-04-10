# 自总：扒模块

## Autoformer 序列分解模块

模块图数据流动：

![segRNN-decomp](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504101939121.png) 

代码摘出：

**使用说明：** 

```bash
输入序列形状: torch.Size([32, 96, 7])
季节性分量形状: torch.Size([32, 96, 7])
趋势性分量形状: torch.Size([32, 96, 7])
```

- 输入输出形状相同
- 在提取趋势性分量时，用的移动平均，同时提取的第一个时间和最后一个时间步进行了 复制填充，保证 移动平均以后输入和输出序列长度保持不变
- 季节性分量直接是 `原始序列 - 趋势性分量` 
- 按照实际情况，修改 `B, L, D  = 32, 96, 7` 和 `kernel_size = 25` 即可
- 1D avgPool 作用在 `dim = -1`

```python
import torch
import torch.nn as nn 
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) # AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))

    def forward(self, x):
        # padding on the both ends of time series

        # 提取第一个时间步并重复，用于前端填充 x.shape =  [32, 36, 7]
        #  [B, L, D] -> [B, 1, D] -> [B, (kernel_size-1)//2, D]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) 

        # 提取最后一个时间步并重复，用于后端填充
        # [B, L, D] -> [B, 1, D] -> [B, (kernel_size-1)//2, D]
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        # 连接填充部分与原序列
        # [B, (k-1)//2, D] + [B, L, D] + [B, (k-1)//2, D] -> [B, L+(k-1), D]
        x = torch.cat([front, x, end], dim=1)

        # 转置并应用一维平均池化
        # [B, L+(k-1), D] -> [B, D, L+(k-1)] -> [B, D, L]
        # 池化窗口大小为kernel_size，步长为1，输出长度为(L+(k-1)-k+1)=L （length + 2P - K + 1）
        x = self.avg(x.permute(0, 2, 1))

        # 转置回原始维度顺序 [B, D, L] -> [B, L, D]
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):

        # 计算移动平均，提取序列趋势分量
        # x 形状[B, L, D] -> moving_mean形状[B, L, D]
        #  moving_avg内部会进行填充，保证输出形状与输入相同
        moving_mean = self.moving_avg(x)

        # 通过原始序列减去趋势分量，得到残差(季节性分量)，逐元素减法操作
        # x形状[B, L, D] - moving_mean形状[B, L, D] -> res形状[B, L, D]
        res = x - moving_mean

        # 返回季节性分量和趋势分量，均保持原始形状[B, L, D]
        # 第一个返回值res是季节性分量，第二个返回值moving_mean是趋势分量
        return res, moving_mean


# init
# Decomp
kernel_size = 25
decomp = series_decomp(kernel_size)

B, L, D  = 32, 96, 7

# forward
x_enc = torch.randn(B, L, D )
# 对输入序列进行时间序列分解，将x_enc[B, L, D]分解为季节性和趋势成分，形状保持不变
seasonal_init, trend_init = decomp(x_enc) # 均为[B, L, D]

print("输入序列形状:",x_enc.shape)
print("季节性分量形状:",seasonal_init.shape)
print("趋势性分量形状:",trend_init.shape)
```

## SegRNN 序列分段

