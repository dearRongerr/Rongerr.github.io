# 查看 pytorch 网络结构

## 法①：`print()`

实例化好模型即可 `print()`

```python
# 实例化模型
model = Model(configs)
print("模型结构:")
print(model)
# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数数量: {total_params}")
```

输出：

<details>
<summary>输出：</summary>
<p>


```python
模型结构:
Model(
  (revin_layer): RevIN()
  (Maxpools): ModuleList(
    (0-3): 4 x AvgPool1d(kernel_size=(3,), stride=(2,), padding=(1,))
  )
  (down_blocks): ModuleList(
    (0): block_model(
      (Linear_channel): ModuleList(
        (0-6): 7 x Linear(in_features=96, out_features=720, bias=True)
      )
      (ln): LayerNorm((720,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU(inplace=True)
    )
    (1): block_model(
      (Linear_channel): ModuleList(
        (0-6): 7 x Linear(in_features=48, out_features=360, bias=True)
      )
      (ln): LayerNorm((360,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU(inplace=True)
    )
    (2): block_model(
      (Linear_channel): ModuleList(
        (0-6): 7 x Linear(in_features=24, out_features=180, bias=True)
      )
      (ln): LayerNorm((180,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU(inplace=True)
    )
    (3): block_model(
      (Linear_channel): ModuleList(
        (0-6): 7 x Linear(in_features=12, out_features=90, bias=True)
      )
      (ln): LayerNorm((90,), eps=1e-05, elementwise_affine=True)
      (relu): ReLU(inplace=True)
    )
  )
  (dct_models): ModuleList(
    (0): dct_channel_block(
      (fc): Sequential(
        (0): Linear(in_features=96, out_features=192, bias=False)
        (1): Dropout(p=0.5, inplace=False)
        (2): ReLU(inplace=True)
        (3): Linear(in_features=192, out_features=96, bias=False)
        (4): Dropout(p=0.5, inplace=False)
        (5): Sigmoid()
      )
      (dct_norm): LayerNorm((96,), eps=1e-06, elementwise_affine=True)
    )
    (1): dct_channel_block(
      (fc): Sequential(
        (0): Linear(in_features=48, out_features=96, bias=False)
        (1): Dropout(p=0.5, inplace=False)
        (2): ReLU(inplace=True)
        (3): Linear(in_features=96, out_features=48, bias=False)
        (4): Dropout(p=0.5, inplace=False)
        (5): Sigmoid()
      )
      (dct_norm): LayerNorm((48,), eps=1e-06, elementwise_affine=True)
    )
    (2): dct_channel_block(
      (fc): Sequential(
        (0): Linear(in_features=24, out_features=48, bias=False)
        (1): Dropout(p=0.5, inplace=False)
        (2): ReLU(inplace=True)
        (3): Linear(in_features=48, out_features=24, bias=False)
        (4): Dropout(p=0.5, inplace=False)
        (5): Sigmoid()
      )
      (dct_norm): LayerNorm((24,), eps=1e-06, elementwise_affine=True)
    )
    (3): dct_channel_block(
      (fc): Sequential(
        (0): Linear(in_features=12, out_features=24, bias=False)
        (1): Dropout(p=0.5, inplace=False)
        (2): ReLU(inplace=True)
        (3): Linear(in_features=24, out_features=12, bias=False)
        (4): Dropout(p=0.5, inplace=False)
        (5): Sigmoid()
      )
      (dct_norm): LayerNorm((12,), eps=1e-06, elementwise_affine=True)
    )
  )
  (icbs): ModuleList(
    (0): ICB(
      (conv1): Conv1d(270, 270, kernel_size=(1,), stride=(1,))
      (conv3): Conv1d(270, 270, kernel_size=(3,), stride=(1,), padding=(1,))
      (conv5): Conv1d(270, 270, kernel_size=(5,), stride=(1,), padding=(2,))
      (conv7): Conv1d(270, 270, kernel_size=(7,), stride=(1,), padding=(3,))
      (conv_out): Conv1d(1080, 180, kernel_size=(1,), stride=(1,))
      (drop): Dropout(p=0.5, inplace=False)
      (act): Tanh()
      (sigmoid): Sigmoid()
      (relu): ReLU()
    )
    (1): ICB(
      (conv1): Conv1d(540, 540, kernel_size=(1,), stride=(1,))
      (conv3): Conv1d(540, 540, kernel_size=(3,), stride=(1,), padding=(1,))
      (conv5): Conv1d(540, 540, kernel_size=(5,), stride=(1,), padding=(2,))
      (conv7): Conv1d(540, 540, kernel_size=(7,), stride=(1,), padding=(3,))
      (conv_out): Conv1d(2160, 360, kernel_size=(1,), stride=(1,))
      (drop): Dropout(p=0.5, inplace=False)
      (act): Tanh()
      (sigmoid): Sigmoid()
      (relu): ReLU()
    )
    (2): ICB(
      (conv1): Conv1d(1080, 1080, kernel_size=(1,), stride=(1,))
      (conv3): Conv1d(1080, 1080, kernel_size=(3,), stride=(1,), padding=(1,))
      (conv5): Conv1d(1080, 1080, kernel_size=(5,), stride=(1,), padding=(2,))
      (conv7): Conv1d(1080, 1080, kernel_size=(7,), stride=(1,), padding=(3,))
      (conv_out): Conv1d(4320, 720, kernel_size=(1,), stride=(1,))
      (drop): Dropout(p=0.5, inplace=False)
      (act): Tanh()
      (sigmoid): Sigmoid()
      (relu): ReLU()
    )
  )
  (channel_mixer): ChebyKANLayer(
    (fc1): ChebyKANLinear()
  )
)
模型参数数量: 29289904
输入形状: torch.Size([16, 96, 7])
输出形状: torch.Size([16, 720, 7])
预期输出形状: [batch_size=16, pred_len=720, enc_in=7]
```
</p>
</details>

## 法②：`torchinfo summary()`

> 注意在同一个设备上

```python
pip install torchinfo
from torchinfo import summary
model = Model(configs)
x = torch.randn(batch_size, seq_len, enc_in)
summary(model, input_data=x)
```



<details>
<summary>输出：</summary>
<p>


```python
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Model                                    [16, 720, 7]              --
├─RevIN: 1-1                             [16, 96, 7]               14
├─ChebyKANLayer: 1-2                     [16, 96, 7]               --
│    └─ChebyKANLinear: 2-1               [1536, 7]                 196
├─ModuleList: 1-12                       --                        (recursive)
│    └─block_model: 2-2                  [16, 7, 720]              1,440
│    │    └─ModuleList: 3-1              --                        488,880
├─ModuleList: 1-13                       --                        (recursive)
│    └─dct_channel_block: 2-3            [16, 7, 96]               1
│    │    └─LayerNorm: 3-2               [16, 7, 96]               192
│    │    └─Sequential: 3-3              [16, 7, 96]               36,864
│    │    └─LayerNorm: 3-4               [16, 7, 96]               (recursive)
├─ModuleList: 1-14                       --                        --
│    └─AvgPool1d: 2-4                    [16, 7, 48]               --
├─ModuleList: 1-12                       --                        (recursive)
│    └─block_model: 2-5                  [16, 7, 360]              720
│    │    └─ModuleList: 3-5              --                        123,480
├─ModuleList: 1-13                       --                        (recursive)
│    └─dct_channel_block: 2-6            [16, 7, 48]               1
│    │    └─LayerNorm: 3-6               [16, 7, 48]               96
│    │    └─Sequential: 3-7              [16, 7, 48]               9,216
│    │    └─LayerNorm: 3-8               [16, 7, 48]               (recursive)
├─ModuleList: 1-14                       --                        --
│    └─AvgPool1d: 2-7                    [16, 7, 24]               --
├─ModuleList: 1-12                       --                        (recursive)
│    └─block_model: 2-8                  [16, 7, 180]              360
│    │    └─ModuleList: 3-9              --                        31,500
├─ModuleList: 1-13                       --                        (recursive)
│    └─dct_channel_block: 2-9            [16, 7, 24]               1
│    │    └─LayerNorm: 3-10              [16, 7, 24]               48
│    │    └─Sequential: 3-11             [16, 7, 24]               2,304
│    │    └─LayerNorm: 3-12              [16, 7, 24]               (recursive)
├─ModuleList: 1-14                       --                        --
│    └─AvgPool1d: 2-10                   [16, 7, 12]               --
├─ModuleList: 1-12                       --                        (recursive)
│    └─block_model: 2-11                 [16, 7, 90]               180
│    │    └─ModuleList: 3-13             --                        8,190
├─ModuleList: 1-13                       --                        (recursive)
│    └─dct_channel_block: 2-12           [16, 7, 12]               1
│    │    └─LayerNorm: 3-14              [16, 7, 12]               24
│    │    └─Sequential: 3-15             [16, 7, 12]               576
│    │    └─LayerNorm: 3-16              [16, 7, 12]               (recursive)
├─ModuleList: 1-14                       --                        --
│    └─AvgPool1d: 2-13                   [16, 7, 6]                --
├─ChebyKANLayer: 1-15                    [16, 90, 7]               (recursive)
│    └─ChebyKANLinear: 2-14              [1440, 7]                 (recursive)
├─ModuleList: 1-16                       --                        --
│    └─ICB: 2-15                         [16, 7, 180]              --
│    │    └─Conv1d: 3-17                 [16, 270, 7]              73,170
│    │    └─Conv1d: 3-18                 [16, 270, 7]              218,970
│    │    └─Conv1d: 3-19                 [16, 270, 7]              364,770
│    │    └─Conv1d: 3-20                 [16, 270, 7]              510,570
│    │    └─Tanh: 3-21                   [16, 270, 7]              --
│    │    └─Sigmoid: 3-22                [16, 270, 7]              --
│    │    └─ReLU: 3-23                   [16, 270, 7]              --
│    │    └─Tanh: 3-24                   [16, 270, 7]              --
│    │    └─Sigmoid: 3-25                [16, 270, 7]              --
│    │    └─ReLU: 3-26                   [16, 270, 7]              --
│    │    └─Tanh: 3-27                   [16, 270, 7]              --
│    │    └─Sigmoid: 3-28                [16, 270, 7]              --
│    │    └─ReLU: 3-29                   [16, 270, 7]              --
│    │    └─Tanh: 3-30                   [16, 270, 7]              --
│    │    └─Sigmoid: 3-31                [16, 270, 7]              --
│    │    └─ReLU: 3-32                   [16, 270, 7]              --
│    │    └─Dropout: 3-33                [16, 1080, 7]             --
│    │    └─Conv1d: 3-34                 [16, 180, 7]              194,580
│    └─ICB: 2-16                         [16, 7, 360]              --
│    │    └─Conv1d: 3-35                 [16, 540, 7]              292,140
│    │    └─Conv1d: 3-36                 [16, 540, 7]              875,340
│    │    └─Conv1d: 3-37                 [16, 540, 7]              1,458,540
│    │    └─Conv1d: 3-38                 [16, 540, 7]              2,041,740
│    │    └─Tanh: 3-39                   [16, 540, 7]              --
│    │    └─Sigmoid: 3-40                [16, 540, 7]              --
│    │    └─ReLU: 3-41                   [16, 540, 7]              --
│    │    └─Tanh: 3-42                   [16, 540, 7]              --
│    │    └─Sigmoid: 3-43                [16, 540, 7]              --
│    │    └─ReLU: 3-44                   [16, 540, 7]              --
│    │    └─Tanh: 3-45                   [16, 540, 7]              --
│    │    └─Sigmoid: 3-46                [16, 540, 7]              --
│    │    └─ReLU: 3-47                   [16, 540, 7]              --
│    │    └─Tanh: 3-48                   [16, 540, 7]              --
│    │    └─Sigmoid: 3-49                [16, 540, 7]              --
│    │    └─ReLU: 3-50                   [16, 540, 7]              --
│    │    └─Dropout: 3-51                [16, 2160, 7]             --
│    │    └─Conv1d: 3-52                 [16, 360, 7]              777,960
│    └─ICB: 2-17                         [16, 7, 720]              --
│    │    └─Conv1d: 3-53                 [16, 1080, 7]             1,167,480
│    │    └─Conv1d: 3-54                 [16, 1080, 7]             3,500,280
│    │    └─Conv1d: 3-55                 [16, 1080, 7]             5,833,080
│    │    └─Conv1d: 3-56                 [16, 1080, 7]             8,165,880
│    │    └─Tanh: 3-57                   [16, 1080, 7]             --
│    │    └─Sigmoid: 3-58                [16, 1080, 7]             --
│    │    └─ReLU: 3-59                   [16, 1080, 7]             --
│    │    └─Tanh: 3-60                   [16, 1080, 7]             --
│    │    └─Sigmoid: 3-61                [16, 1080, 7]             --
│    │    └─ReLU: 3-62                   [16, 1080, 7]             --
│    │    └─Tanh: 3-63                   [16, 1080, 7]             --
│    │    └─Sigmoid: 3-64                [16, 1080, 7]             --
│    │    └─ReLU: 3-65                   [16, 1080, 7]             --
│    │    └─Tanh: 3-66                   [16, 1080, 7]             --
│    │    └─Sigmoid: 3-67                [16, 1080, 7]             --
│    │    └─ReLU: 3-68                   [16, 1080, 7]             --
│    │    └─Dropout: 3-69                [16, 4320, 7]             --
│    │    └─Conv1d: 3-70                 [16, 720, 7]              3,111,120
├─RevIN: 1-17                            [16, 720, 7]              (recursive)
==========================================================================================
Total params: 29,289,904
Trainable params: 29,289,904
Non-trainable params: 0
Total mult-adds (G): 3.21
==========================================================================================
Input size (MB): 0.04
Forward/backward pass size (MB): 10.82
Params size (MB): 117.15
Estimated Total Size (MB): 128.01
==========================================================================================
```

</p>
</details>


## 法③：torchsummary summary()

```python
from torchsummary import summary
# 实例化模型
model = Model(configs)
# 生成随机输入数据
batch_size = 16
seq_len = configs.seq_len
enc_in = configs.enc_in
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(batch_size, seq_len, enc_in).to(device)
model = model.to(device)
# summary(model, input_data=x)
summary(model, input_size=(configs.seq_len, configs.enc_in), batch_size=batch_size, device=device.type)
```


<details>
<summary>输出：</summary>
<p>


```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
             RevIN-1                [16, 96, 7]               0
    ChebyKANLinear-2                    [16, 7]               0
     ChebyKANLayer-3                [16, 96, 7]               0
            Linear-4                  [16, 720]          69,840
            Linear-5                  [16, 720]          69,840
            Linear-6                  [16, 720]          69,840
            Linear-7                  [16, 720]          69,840
            Linear-8                  [16, 720]          69,840
            Linear-9                  [16, 720]          69,840
           Linear-10                  [16, 720]          69,840
      block_model-11               [16, 7, 720]               0
        LayerNorm-12                [16, 7, 96]             192
           Linear-13               [16, 7, 192]          18,432
          Dropout-14               [16, 7, 192]               0
             ReLU-15               [16, 7, 192]               0
           Linear-16                [16, 7, 96]          18,432
          Dropout-17                [16, 7, 96]               0
          Sigmoid-18                [16, 7, 96]               0
        LayerNorm-19                [16, 7, 96]             192
dct_channel_block-20                [16, 7, 96]               0
        AvgPool1d-21                [16, 7, 48]               0
           Linear-22                  [16, 360]          17,640
           Linear-23                  [16, 360]          17,640
           Linear-24                  [16, 360]          17,640
           Linear-25                  [16, 360]          17,640
           Linear-26                  [16, 360]          17,640
           Linear-27                  [16, 360]          17,640
           Linear-28                  [16, 360]          17,640
      block_model-29               [16, 7, 360]               0
        LayerNorm-30                [16, 7, 48]              96
           Linear-31                [16, 7, 96]           4,608
          Dropout-32                [16, 7, 96]               0
             ReLU-33                [16, 7, 96]               0
           Linear-34                [16, 7, 48]           4,608
          Dropout-35                [16, 7, 48]               0
          Sigmoid-36                [16, 7, 48]               0
        LayerNorm-37                [16, 7, 48]              96
dct_channel_block-38                [16, 7, 48]               0
        AvgPool1d-39                [16, 7, 24]               0
           Linear-40                  [16, 180]           4,500
           Linear-41                  [16, 180]           4,500
           Linear-42                  [16, 180]           4,500
           Linear-43                  [16, 180]           4,500
           Linear-44                  [16, 180]           4,500
           Linear-45                  [16, 180]           4,500
           Linear-46                  [16, 180]           4,500
      block_model-47               [16, 7, 180]               0
        LayerNorm-48                [16, 7, 24]              48
           Linear-49                [16, 7, 48]           1,152
          Dropout-50                [16, 7, 48]               0
             ReLU-51                [16, 7, 48]               0
           Linear-52                [16, 7, 24]           1,152
          Dropout-53                [16, 7, 24]               0
          Sigmoid-54                [16, 7, 24]               0
        LayerNorm-55                [16, 7, 24]              48
dct_channel_block-56                [16, 7, 24]               0
        AvgPool1d-57                [16, 7, 12]               0
           Linear-58                   [16, 90]           1,170
           Linear-59                   [16, 90]           1,170
           Linear-60                   [16, 90]           1,170
           Linear-61                   [16, 90]           1,170
           Linear-62                   [16, 90]           1,170
           Linear-63                   [16, 90]           1,170
           Linear-64                   [16, 90]           1,170
      block_model-65                [16, 7, 90]               0
        LayerNorm-66                [16, 7, 12]              24
           Linear-67                [16, 7, 24]             288
          Dropout-68                [16, 7, 24]               0
             ReLU-69                [16, 7, 24]               0
           Linear-70                [16, 7, 12]             288
          Dropout-71                [16, 7, 12]               0
          Sigmoid-72                [16, 7, 12]               0
        LayerNorm-73                [16, 7, 12]              24
dct_channel_block-74                [16, 7, 12]               0
        AvgPool1d-75                 [16, 7, 6]               0
   ChebyKANLinear-76                    [16, 7]               0
    ChebyKANLayer-77                [16, 90, 7]               0
           Conv1d-78               [16, 270, 7]          73,170
           Conv1d-79               [16, 270, 7]         218,970
           Conv1d-80               [16, 270, 7]         364,770
           Conv1d-81               [16, 270, 7]         510,570
             Tanh-82               [16, 270, 7]               0
          Sigmoid-83               [16, 270, 7]               0
             ReLU-84               [16, 270, 7]               0
             Tanh-85               [16, 270, 7]               0
          Sigmoid-86               [16, 270, 7]               0
             ReLU-87               [16, 270, 7]               0
             Tanh-88               [16, 270, 7]               0
          Sigmoid-89               [16, 270, 7]               0
             ReLU-90               [16, 270, 7]               0
             Tanh-91               [16, 270, 7]               0
          Sigmoid-92               [16, 270, 7]               0
             ReLU-93               [16, 270, 7]               0
          Dropout-94              [16, 1080, 7]               0
           Conv1d-95               [16, 180, 7]         194,580
              ICB-96               [16, 7, 180]               0
           Conv1d-97               [16, 540, 7]         292,140
           Conv1d-98               [16, 540, 7]         875,340
           Conv1d-99               [16, 540, 7]       1,458,540
          Conv1d-100               [16, 540, 7]       2,041,740
            Tanh-101               [16, 540, 7]               0
         Sigmoid-102               [16, 540, 7]               0
            ReLU-103               [16, 540, 7]               0
            Tanh-104               [16, 540, 7]               0
         Sigmoid-105               [16, 540, 7]               0
            ReLU-106               [16, 540, 7]               0
            Tanh-107               [16, 540, 7]               0
         Sigmoid-108               [16, 540, 7]               0
            ReLU-109               [16, 540, 7]               0
            Tanh-110               [16, 540, 7]               0
         Sigmoid-111               [16, 540, 7]               0
            ReLU-112               [16, 540, 7]               0
         Dropout-113              [16, 2160, 7]               0
          Conv1d-114               [16, 360, 7]         777,960
             ICB-115               [16, 7, 360]               0
          Conv1d-116              [16, 1080, 7]       1,167,480
          Conv1d-117              [16, 1080, 7]       3,500,280
          Conv1d-118              [16, 1080, 7]       5,833,080
          Conv1d-119              [16, 1080, 7]       8,165,880
            Tanh-120              [16, 1080, 7]               0
         Sigmoid-121              [16, 1080, 7]               0
            ReLU-122              [16, 1080, 7]               0
            Tanh-123              [16, 1080, 7]               0
         Sigmoid-124              [16, 1080, 7]               0
            ReLU-125              [16, 1080, 7]               0
            Tanh-126              [16, 1080, 7]               0
         Sigmoid-127              [16, 1080, 7]               0
            ReLU-128              [16, 1080, 7]               0
            Tanh-129              [16, 1080, 7]               0
         Sigmoid-130              [16, 1080, 7]               0
            ReLU-131              [16, 1080, 7]               0
         Dropout-132              [16, 4320, 7]               0
          Conv1d-133               [16, 720, 7]       3,111,120
             ICB-134               [16, 7, 720]               0
           RevIN-135               [16, 720, 7]               0
================================================================
Total params: 29,287,350
Trainable params: 29,287,350
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.04
Forward/backward pass size (MB): 39.54
Params size (MB): 111.72
Estimated Total Size (MB): 151.30
----------------------------------------------------------------
输入形状: torch.Size([16, 96, 7])
```

</p>
</details>
