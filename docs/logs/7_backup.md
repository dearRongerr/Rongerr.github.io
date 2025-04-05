# Backup

## vscode 

### launch.json

```python
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
    
        {
            "name": "Python: run_longExp.py",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/run_longExp.py",
            "args": [
                "--model_id", "illness_60_24",
                "--is_training", "1" ,
                "--model", "SegRNN", 
                "--data", "custom",
                "--root_path", "./dataset/",
                "--data_path", "national_illness.csv",
                "--features", "M",
                "--seq_len", "60",
                "--pred_len", "24",
                "--d_model", "512",
                "--dropout", "0.0",
                "--rnn_type", "gru",
                "--dec_way", "pmf",
                "--seg_len", "12",
                "--loss", "mae",
                "--des", "test",
                "--itr", "1",
                "--batch_size" ,"16",
                "--train_epochs", "2",
                "--num_workers", "0",
                "--learning_rate", "0.001",
                "--enc_in", "7",
                "--revin", "1"
            ],
            "console": "integratedTerminal",
            "justMyCode": true,
            "cwd": "${workspaceFolder}"
        },


        
        {
            "name": "[这里更换为任意名称]",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5998
            }
        },
        {
            "type": "bashdb",
            "request": "launch",
            "name": "Bash-Debug (type in script name)",
            "cwd": "${workspaceFolder}",
            "program": "${command:AskForScriptName}",
            "args": []
        },
        {
            "type": "bashdb",
            "request": "launch",
            "name": "Bash-Debug (select script from list of sh files)",
            "cwd": "${workspaceFolder}",
            "program": "${command:SelectScriptName}",
            "args": []
        },
        {
            "type": "bashdb",
            "request": "launch",
            "name": "Bash-Debug (hardcoded script name)",
            "cwd": "${workspaceFolder}",
            "program": "${workspaceFolder}/path/to/script.sh",
            "args": []
        },
        {
            "type": "bashdb",
            "request": "launch",
            "name": "Bash-Debug (simplest configuration)",
            "program": "${file}"
        }
    ]
}
```



### sh

```python
model_name=SegRNN

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=illness
data_name=custom


seq_len=60
for pred_len in 24 36 48 60
do
    python -m debugpy --listen 5998 --wait-for-client run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len 12 \
      --enc_in 7 \
      --d_model 512 \
      --dropout 0 \
      --train_epochs 30 \
      --patience 10 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --revin 1 \
      --itr 1 --batch_size 16 --learning_rate 0.001
done
```

## 傻瓜式实现UnetTSF

<details>
<summary>UnetTSF 不涉及 for循环</summary>
<p>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

"UnetTSF: A Better Performance Linear Complexity Time Series Prediction Model"


class block_model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, input_channels, input_len, out_len):
        super(block_model, self).__init__()
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len

        self.Linear_channel = nn.Linear(self.input_len, self.out_len)
        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # (B,C,N,T) --> (B,C,N,T)
        output = self.Linear_channel(x)
        return output



class Model(nn.Module):
    def __init__(self, input_channels=64,out_channels=64, seq_len=720, pred_len=720):
        super(Model, self).__init__()

        self.input_channels = input_channels
        self.out_channels = out_channels
        self.input_len = seq_len
        self.out_len = pred_len

        # 下采样设定
        n1 = 1
        # 序列长度要能够被下采样倍数整除
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        # 当输入序列长度等于40时候, down_in=[40,20,10,5]; down_out=[40,20,10,5]
        down_in = [int(self.input_len / filters[0]), int(self.input_len / filters[1]), int(self.input_len / filters[2]),int(self.input_len / filters[3])]
        down_out = [int(self.out_len / filters[0]), int(self.out_len / filters[1]), int(self.out_len / filters[2]),int(self.out_len / filters[3])]

        # 最大池化层
        # out = (input+2*padding-kernelsize)/stride + 1  在这里配置一样,因此简化为: out=(input+2-3)/2+1=(input+1)/2
        self.Maxpool1 = nn.AvgPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.Maxpool2 = nn.AvgPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.Maxpool3 = nn.AvgPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.Maxpool4 = nn.AvgPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        # 左边特征提取层
        self.down_block1 = block_model(self.input_channels, down_in[0], down_out[0])
        self.down_block2 = block_model(self.input_channels, down_in[1], down_out[1])
        self.down_block3 = block_model(self.input_channels, down_in[2], down_out[2])
        self.down_block4 = block_model(self.input_channels, down_in[3], down_out[3])

        # 右边特征融合层
        self.up_block3 = block_model(self.input_channels, down_out[2] + down_out[3], down_out[2])
        self.up_block2 = block_model(self.input_channels, down_out[1] + down_out[2], down_out[1])
        self.up_block1 = block_model(self.input_channels, down_out[0] + down_out[1], down_out[0])

        # 输出映射
        self.linear_out = nn.Linear(self.input_channels, self.out_channels)

    def forward(self, x):
        # x: (N,T,C), 下面的注释我们按照43行提供的采用率来进行计算

        x1 = x.permute(2,0,1)    # 对输入x变换维度,便于计算:(N,T,C) --> (C,N,T)
        e1 = self.down_block1(x1)  # 通过线性层映射到中间特征表示E1: (C,N,T) --> (C,N,T)

        x2 = self.Maxpool1(x1)    # 对输入x1通过池化操作获得第二层特征x2: (C,N,T)--> (C,N,T/2)
        e2 = self.down_block2(x2) # 通过线性层映射到中间特征表示E2:(C,N,T/2)--> (C,N,T/2)

        x3 = self.Maxpool2(x2)     # 对输入x2通过池化操作获得第二层特征x3:(C,N,T/2)-->(B,C,N,T/4)
        e3 = self.down_block3(x3)  #通过线性层映射到中间特征表示E3:(B,C,N,T/4)-->(B,C,N,T/4)

        x4 = self.Maxpool3(x3)    #对输入x3通过池化操作获得第二层特征x4:(B,C,N,T/4)-->(B,C,N,T/8)
        e4 = self.down_block4(x4) #通过线性层映射到中间特征表示E4:(B,C,N,T/8) --> (B,C,N,T/8)


        # 第四层向第三层融合
        d3 = torch.cat((e3, e4), dim=-1)  # 将e3特征图与d4==E4特征图在时间维度上拼接:  (B,C,N,T/4).concat(B,C,N,T/8) == (B,C,N,3T/8)
        d3 = self.up_block3(d3) # 将拼接后的时间维度重新映射到当前层本该具有的时间维度: (B,C,N,3T/8) --> (B,C,N,T/4)

        # 第三层向第二层融合
        d2 = torch.cat((e2, d3), dim=-1)  # 将e2特征图与d3特征图在时间维度上拼接: (B,C,N,T/2).concat(B,C,N,T/4) == (B,C,N,3T/4)
        d2 = self.up_block2(d2)  # 将拼接后的时间维度重新映射到当前层本该具有的时间维度:(B,C,N,3T/4)--> (B,C,N,T/2)

        # 第二层向第一层融合
        d1 = torch.cat((e1, d2), dim=-1)  # 将e1特征图与d1特征图时间维度上拼接: (B,C,N,T).concat(B,C,N,T/2) == (B,C,N,3T/2)
        out = self.up_block1(d1) # 将拼接后的时间维度重新映射到当前层本该具有的时间维度:(B,C,N,3T/2)--> (B,C,N,T)

        out = self.linear_out(out.permute(1,2,0)) # 将第一层融合后的表示通过一个线性层生成输出: (B,C 0 ,N 1,T 2 )--permute-> (B,N 1,T 2,C 0)--linear->(B,N,T,C)

        return out

if __name__ == '__main__':
    # (B,N,T,C)  T:序列的长度,N:序列的个数;  更改输入记得把104行的通道,输入输出长度相应的改变; 输入的序列长度要被第45行的下采样比例整除才行。
    X = torch.randn( 16, 720, 321)
    Model = Model(input_channels=64,out_channels=64, seq_len=40, pred_len=40)
    out = Model(X) # (B,N,T,C)-->(B,N,T,C)
    print(out.shape)
    
```
</p>
</details>

