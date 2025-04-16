# UNetTSF

> 开始部分见 2025-04-16 Wednesday ，okay，继续

`block_model`

```python 
if self.individual:
    self.Linear_channel = nn.ModuleList()

    for i in range(self.channels):
        self.Linear_channel.append(nn.Linear(self.input_len, self.out_len))
else:
    self.Linear_channel = nn.Linear(self.input_len, self.out_len)
self.ln = nn.LayerNorm(out_len)
self.relu = nn.ReLU(inplace=True)
```

通过 individual 控制初始化几个线性层，判断是通道独立还是共享参数，这里就全是库函数，不用再扒了。

其实原始 block 应该是 96→720 这样，这里（好像）有点问题。

查了一下：`self.Linear_channel`

```python
Linear(in_features=96, out_features=192, bias=True)
```

emm，好像是一样的。看下面，进行趋势分解：

```python
trend, cyclic = self.decomposer(base_output)
```

不看初始化了，直接查这个东西是啥：

```
self.decomposer
TrendCyclicDecomposition(
  (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(12,))
)
```

- padding & kernel size，很明显就是一个 移动平均。
- 第一个返回值：趋势项，第二个返回值：残差项
- 接下来 进入 forward

```python
    def forward(self, x):
        """
        输入 x: [B, C, L]
        输出: trend [B, C, L], cyclic [B, C, L]
        """
        # 提取趋势项(通过移动平均)
        trend = self.avg(x)
        
        # 计算周期项(原始信号减去趋势)
        cyclic = x - trend
        
        return trend, cyclic
```

这里的 forward 也挺简单的

一个 self.avg，参数

```python
AvgPool1d(kernel_size=(25,), stride=(1,), padding=(12,))
```

也没啥好说的，上面查 `self.decomposer` 也能看到，具体来说就是对 `AvgPool1d` 的实例化。

- 到了库函数不用步进了
- 下面就是 原始序列 减去 趋势项 得到季节性成分，返回。

```python
trend, cyclic = self.decomposer(base_output)
```

输入输出，形状都是：BCL（通道优先的格式）

```python
freq_bands = self.freq_decomposer(cyclic)
```

接下来处理，高频成分，也就是季节性成分。

查一下：`self.freq_decomposer`  这是啥东西

```python
FrequencyDecomposer(
  (wavelet_decomposers): ModuleList(
    (0): SimpleWaveletDecomposition(
      (low_pass): Sequential(
        (0): ReflectionPad1d((2, 1))
        (1): Conv1d(7, 7, kernel_size=(4,), stride=(1,), groups=7)
        (2): GELU()
        (3): InstanceNorm1d(7, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
      (high_pass): Sequential(
        (0): ReflectionPad1d((2, 1))
        (1): Conv1d(7, 7, kernel_size=(4,), stride=(1,), groups=7)
        (2): GELU()
        (3): InstanceNorm1d(7, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (1): SimpleWaveletDecomposition(
      (low_pass): Sequential(
        (0): ReflectionPad1d((2, 1))
        (1): Conv1d(7, 7, kernel_size=(4,), stride=(1,), groups=7)
        (2): GELU()
        (3): InstanceNorm1d(7, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
      (high_pass): Sequential(
        (0): ReflectionPad1d((2, 1))
        (1): Conv1d(7, 7, kernel_size=(4,), stride=(1,), groups=7)
        (2): GELU()
        (3): InstanceNorm1d(7, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
  )
)
```

内容有点多，但已经好多了，一点点扒拉把

- self.freq_decomposer 是实例化的 FrequencyDecomposer 这个类
-  FrequencyDecomposer 这个类里面第一个是 `(wavelet_decomposers)`

- (wavelet_decomposers)这个东西是 ModuleList 的实例化
- ModuleList类里面是两个  (0): SimpleWaveletDecomposition  和 (1): SimpleWaveletDecomposition 既然如此，只看一个
-  (0): SimpleWaveletDecomposition是 (low_pass) 和 (high_pass) 的实例化
-  (low_pass) 和 (high_pass) 分别是两个 Sequential 的实例化
- 到库函数了，不用扒了
- 如果仔细看的话，(low_pass) 和 (high_pass)  低通和高通两个 内部是一样的
- 拿出来看看

```python
(low_pass): Sequential(
(0): ReflectionPad1d((2, 1))
(1): Conv1d(7, 7, kernel_size=(4,), stride=(1,), groups=7)
(2): GELU()
(3): InstanceNorm1d(7, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
)
(high_pass): Sequential(
(0): ReflectionPad1d((2, 1))
(1): Conv1d(7, 7, kernel_size=(4,), stride=(1,), groups=7)
(2): GELU()
(3): InstanceNorm1d(7, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
)
```

- 画个图把：

![image-20250416171956882](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504161720706.png)

还好了，橙色表示是一样的东西。