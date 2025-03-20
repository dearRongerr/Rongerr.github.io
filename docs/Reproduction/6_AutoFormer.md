# Autoformer

## github 源码主页

Autoformer (NeurIPS 2021) 自动成型机 (NeurIPS 2021)

Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
Autoformer：用于长期序列预测的具有自相关的分解变压器

Time series forecasting is a critical demand for real applications. Enlighted by the classic time series analysis and stochastic process theory, we propose the Autoformer as a general series forecasting model [[paper](https://arxiv.org/abs/2106.13008)]. **Autoformer goes beyond the Transformer family and achieves the series-wise connection for the first time.**
时间序列预测是实际应用的关键需求。受经典时间序列分析和随机过程理论的启发，我们提出了 Autoformer 作为通用序列预测模型 [[论文](https://arxiv.org/abs/2106.13008)]。Autoformer**超越了 Transformer 家族，首次实现了序列连接。**

In long-term forecasting, Autoformer achieves SOTA, with a **38% relative improvement** on six benchmarks, covering five practical applications: **energy, traffic, economics, weather and disease**.
在长期预测中，Autoformer 实现了 SOTA，在六个基准上**相对提升了 38%** ，涵盖了**能源、交通、经济、天气和疾病**五个实际应用。

**News** (2023.08) Autoformer has been included in [Hugging Face](https://huggingface.co/models?search=autoformer). See [blog](https://huggingface.co/blog/autoformer).
🚩**新闻**(2023.08) Autoformer 已包含在[Hugging Face](https://huggingface.co/models?search=autoformer)中。查看[博客](https://huggingface.co/blog/autoformer)。

🚩**News** (2023.06) The extension version of Autoformer ([Interpretable weather forecasting for worldwide stations with a unified deep model](https://www.nature.com/articles/s42256-023-00667-9)) has been published in Nature Machine Intelligence as the [Cover Article](https://www.nature.com/natmachintell/volumes/5/issues/6).
🚩**新闻**(2023.06) Autoformer 的扩展版本 ([使用统一深度模型为全球站点提供可解释的天气预报](https://www.nature.com/articles/s42256-023-00667-9)) 在《自然机器智能》杂志上作为[封面文章](https://www.nature.com/natmachintell/volumes/5/issues/6)发表。

🚩**News** (2023.02) Autoformer has been included in our [[Time-Series-Library\]](https://github.com/thuml/Time-Series-Library), which covers long- and short-term forecasting, imputation, anomaly detection, and classification.
🚩**新闻**(2023.02) Autoformer 已包含在我们的[[时间序列库\]](https://github.com/thuml/Time-Series-Library)中，它涵盖长期和短期预测、归纳、异常检测和分类。

🚩**News** (2022.02-2022.03) Autoformer has been deployed in [2022 Winter Olympics](https://en.wikipedia.org/wiki/2022_Winter_Olympics) to provide weather forecasting for competition venues, including wind speed and temperature.
🚩**新闻**（2022.02-2022.03）Autoformer 已部署在[2022 年冬奥会，](https://en.wikipedia.org/wiki/2022_Winter_Olympics)为比赛场馆提供天气预报，包括风速、温度等。

## 准备

### git clone

![image-20250317144505215](images/image-20250317144505215.png) 

克隆远程仓库的方法：

（1）HTTPS，在把本地仓库的代码 push 到远程仓库的时候，需要验证用户名和密码

（2）SSH，git 开头的是 SSH 协议，这种方式在推送的时候，不需要验证用户名和密码，但是需要在 github 上添加SSH公钥的配置（推荐）

（3）zip download

我这里使用了 SSH 配置：

![image-20250317144903028](images/image-20250317144903028.png) 

服务器直接 git clone 是很慢。所以本地 git clone，然后再上传服务器。

![image-20250317145242243](images/image-20250317145242243.png) 

本地下载好以后，使用 FileZilla上传到远程服务器

![image-20250317145427044](images/image-20250317145427044.png)  

down到本地以后，删除 .git文件，取消连接着远程仓库

![image-20250317145705700](images/image-20250317145705700.png) 

![image-20250317145752968](images/image-20250317145752968.png) 

### readme

下载数据集

设置数据集路径

![image-20250317150739551](images/image-20250317150739551.png) 

### 调试配置

新建配置文件

![image-20250317150852423](images/image-20250317150852423.png) 



修改配置文件

![image-20250317151048416](images/image-20250317151048416.png)  



修改配置文件

```
        {
            "name": "Autoformer",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5997
            }
        },
```

修改 sh 文件

```
python -m debugpy --listen 5997 --wait-for-client run.py \
```

### 新建 python 虚拟环境

本实验所需要的实验环境

> Install Python 3.6, PyTorch 1.9.0.

参考命令

```
conda create -n dave python==3.8
conda activate dave
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy
conda install scikit-image
conda install scikit-learn
conda install tqdm
conda install pycocotools
```

激活、退出：

```
# To activate this environment, use               
#     $ conda activate Autoformer
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

用 requirements.txt 安装需要的库

```
conda create -n SegRNN python=3.8
conda activate SegRNN
pip install -r requirements.txt
```

启动 sh 文件：

```
sh run_main.sh
```

**适用于本实验的所有命令 :**

```
conda create -n Autoformer python=3.6
conda activate Autoformer
```

[pytorch 官网](https://pytorch.org/)查看所需命令

![image-20250317154037815](images/image-20250317154037815.png) 



![image-20250317153952283](images/image-20250317153952283.png)



```
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
```

### requirements

```
pip install -r requirements.txt
```

或者：

```
conda create -n Autoformer python=3.6
conda activate Autoformer
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
conda install pandas
conda install scikit-learn
conda install debugpy
conda install matplotlib
conda install reformer_pytorch
```

配置好以后，成功进入调试：

![image-20250317155629524](images/image-20250317155629524.png)  



## 开始调试

代码相似度极高。

**Autoformer init：36（18）-》24**

![image-20250317160059801](images/image-20250317160059801.png)

setting:

```
ili_36_24_Autoformer_custom_ftM_sl36_ll18_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0
```

model_id  36 预测 24 步长（label=18）、AutoFormer 模型，自定义数据集，预测多变量，输入序列 36，标签序列 18，预测序列 24，嵌入维度 512，注意力头数 8，2层编码层，1 层解码层，

```
df2048_fc3_ebtimeF_dtTrue_Exp_0
         		args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
```

**Autoformer model**

```python
Model(
  (decomp): series_decomp(
    (moving_avg): moving_avg(
      (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
    )
  )
  (enc_embedding): DataEmbedding_wo_pos(
    (value_embedding): TokenEmbedding(
      (tokenConv): Conv1d(7, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False, padding_mode=circular)
    )
    (position_embedding): PositionalEmbedding()
    (temporal_embedding): TimeFeatureEmbedding(
      (embed): Linear(in_features=4, out_features=512, bias=False)
    )
    (dropout): Dropout(p=0.05, inplace=False)
  )
  (dec_embedding): DataEmbedding_wo_pos(
    (value_embedding): TokenEmbedding(
      (tokenConv): Conv1d(7, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False, padding_mode=circular)
    )
    (position_embedding): PositionalEmbedding()
    (temporal_embedding): TimeFeatureEmbedding(
      (embed): Linear(in_features=4, out_features=512, bias=False)
    )
    (dropout): Dropout(p=0.05, inplace=False)
  )
  (encoder): Encoder(
    (attn_layers): ModuleList(
      (0): EncoderLayer(
        (attention): AutoCorrelationLayer(
          (inner_correlation): AutoCorrelation(
            (dropout): Dropout(p=0.05, inplace=False)
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,), bias=False)
        (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,), bias=False)
        (decomp1): series_decomp(
          (moving_avg): moving_avg(
            (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
          )
        )
        (decomp2): series_decomp(
          (moving_avg): moving_avg(
            (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
          )
        )
        (dropout): Dropout(p=0.05, inplace=False)
      )
      (1): EncoderLayer(
        (attention): AutoCorrelationLayer(
          (inner_correlation): AutoCorrelation(
            (dropout): Dropout(p=0.05, inplace=False)
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,), bias=False)
        (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,), bias=False)
        (decomp1): series_decomp(
          (moving_avg): moving_avg(
            (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
          )
        )
        (decomp2): series_decomp(
          (moving_avg): moving_avg(
            (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
          )
        )
        (dropout): Dropout(p=0.05, inplace=False)
      )
    )
    (norm): my_Layernorm(
      (layernorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
  )
  (decoder): Decoder(
    (layers): ModuleList(
      (0): DecoderLayer(
        (self_attention): AutoCorrelationLayer(
          (inner_correlation): AutoCorrelation(
            (dropout): Dropout(p=0.05, inplace=False)
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (cross_attention): AutoCorrelationLayer(
          (inner_correlation): AutoCorrelation(
            (dropout): Dropout(p=0.05, inplace=False)
          )
          (query_projection): Linear(in_features=512, out_features=512, bias=True)
          (key_projection): Linear(in_features=512, out_features=512, bias=True)
          (value_projection): Linear(in_features=512, out_features=512, bias=True)
          (out_projection): Linear(in_features=512, out_features=512, bias=True)
        )
        (conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,), bias=False)
        (conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,), bias=False)
        (decomp1): series_decomp(
          (moving_avg): moving_avg(
            (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
          )
        )
        (decomp2): series_decomp(
          (moving_avg): moving_avg(
            (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
          )
        )
        (decomp3): series_decomp(
          (moving_avg): moving_avg(
            (avg): AvgPool1d(kernel_size=(25,), stride=(1,), padding=(0,))
          )
        )
        (dropout): Dropout(p=0.05, inplace=False)
        (projection): Conv1d(512, 7, kernel_size=(3,), stride=(1,), padding=(1,), bias=False, padding_mode=circular)
      )
    )
    (norm): my_Layernorm(
      (layernorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (projection): Linear(in_features=512, out_features=7, bias=True)
  )
)
```



数据集的加载是完全一样的。





### 编码器

目的：结合时间特征，将 数据特征嵌入到指定维度

```python
enc_out = self.enc_embedding(x_enc, x_mark_enc)
```



```python
self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,configs.dropout)
```



![image-20250317204752276](images/image-20250317204752276.png)



![image-20250317205258626](images/image-20250317205258626.png)



**流程图**

```python
输入:
x_enc [B, L, D]        x_mark_enc [B, L, time_features]
    |                        |
    v                        v
+-----------------------------------------------+
|           Model.forward()调用                  |
|      self.enc_embedding(x_enc, x_mark_enc)    |
+-----------------------------------------------+
            |                |
            v                v
+------------------------+  +---------------------------+
| TokenEmbedding (值嵌入) |  | TemporalEmbedding (时间嵌入)|
+------------------------+  +---------------------------+
| 输入: x [B, L, D]      |  | 输入: x_mark [B, L, time_f]|
|                        |  |                           |
| 操作:                  |  | 操作:                     |
| 1.转置: [B, D, L]      |  | 1.转换为long类型          |
| 2.1D卷积: D -> d_model |  | 2.提取时间特征:           |
| 3.转置回: [B, L, d_model]|  |   - month_x (x[:,:,0])   |
|                        |  |   - day_x (x[:,:,1])      |
| 输出: [B, L, d_model]  |  |   - weekday_x (x[:,:,2])  |
|                        |  |   - hour_x (x[:,:,3])     |
+------------------------+  |   - minute_x (可选)       |
            |               |                           |
            |               | 3.查表获取各时间特征的嵌入  |
            |               | 4.将所有时间嵌入相加       |
            |               |                           |
            |               | 输出: [B, L, d_model]     |
            |               +---------------------------+
            |                        |
            +------------+------------+
                         v
            +---------------------------+
            | 相加并应用Dropout         |
            | value_emb + temporal_emb |
            +---------------------------+
                         |
                         v
                  输出: enc_out
                 [B, L, d_model]
```



1. **值嵌入 (TokenEmbedding)**:
   - 通过卷积操作将原始特征 [B, L, D] 映射到更高维度表示 [B, L, d_model]
   - 使用循环填充的1D卷积捕获局部特征模式
2. **时间嵌入 (TemporalEmbedding)**:
   - 将时间标记 [B, L, time_features] 转换为 [B, L, d_model] 的嵌入向量
   - 分别为月、日、星期、小时等时间特征查表获取嵌入，然后相加
   - 时间嵌入帮助模型识别时间模式(季节性、每日/每周周期等)
3. **组合嵌入**:
   - 将值嵌入和时间嵌入相加，形成最终编码器输入 [B, L, d_model]
   - 注意此版本不包含位置嵌入(DataEmbedding_wo_pos)

这种多重嵌入方式使模型能同时利用时间序列的值信息和时间特征信息，为后续的注意力机制和时间序列建模提供丰富的上下文。

## 模型定义





### 编码器 解码器部分



```mermaid
classDiagram
    class Model {
        +DataEmbedding_wo_pos enc_embedding
        +DataEmbedding_wo_pos dec_embedding
        +Encoder encoder
        +Decoder decoder
        +series_decomp decomp
        +forward(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
    }
    
    class Encoder {
        +List~EncoderLayer~ layers
        +my_Layernorm norm_layer
        +forward(x, attn_mask)
    }
    
    class EncoderLayer {
        +AutoCorrelationLayer attention
        +Conv1d conv1
        +Conv1d conv2
        +series_decomp decomp1
        +series_decomp decomp2
        +Dropout dropout
        +activation
        +forward(x, attn_mask)
    }
    
    class AutoCorrelationLayer {
        +AutoCorrelation attention
        +Linear query_projection
        +Linear key_projection
        +Linear value_projection
        +Linear out_projection
        +forward(queries, keys, values, attn_mask)
    }
    
    class AutoCorrelation {
        +bool mask_flag
        +int factor
        +float scale
        +Dropout dropout
        +bool output_attention
        +time_delay_agg_training(values, corr)
        +time_delay_agg_inference(values, corr)
        +forward(queries, keys, values, attn_mask)
    }
    
    class Decoder {
        +List~DecoderLayer~ layers
        +my_Layernorm norm_layer
        +Linear projection
        +forward(x, enc_out, x_mask, cross_mask, trend)
    }
    
    class DecoderLayer {
        +AutoCorrelationLayer self_attention
        +AutoCorrelationLayer cross_attention
        +Conv1d conv1
        +Conv1d conv2
        +series_decomp decomp1
        +series_decomp decomp2
        +Dropout dropout
        +activation
        +forward(x, enc_out, x_mask, cross_mask, trend)
    }
    
    Model --> Encoder
    Model --> Decoder
    Encoder --> EncoderLayer
    EncoderLayer --> AutoCorrelationLayer
    EncoderLayer --> Conv1d
    EncoderLayer --> series_decomp
    AutoCorrelationLayer --> AutoCorrelation
    Decoder --> DecoderLayer
    DecoderLayer --> AutoCorrelationLayer
    DecoderLayer --> Conv1d
    DecoderLayer --> series_decomp
```

## 训练过程，形状变换

（1）

代码：

![image-20250319202142537](images/image-20250319202142537.png) 

逐字讲解：

model 训练从 exp_main.py的 train 函数开始，epoch 表示整个训练集迭代几次，for batchx、batchy、batch x mark、batch y mark 一个批次一个批次的训练，第一个 for 训练的 epoch 是我们自己可以设置的，第二个 for 训练的 iteration 迭代次数是 `数据集长度 ➗ batch size`

接下来，调用 `self._predict` 方法进行预测，这里 predict 函数需要的参数 batchx、batchy、batch x mark、batch y mark 形状分别是 `batch_x = [32,36,7], batch_y = [32,42(18+24),7],batch_x_mark=[32,36,4],batch_y_mark = [32,42,4]`

32 表示 一个 batch 样本的个数；

36 表示每个样本的时间步，也可以说是回溯窗口的大小，或者叫输入序列的长度

7 表示 illness 数据集的特征数

batchy 的 42 表示 18 的 label length，是取的 原始输入序列的 二分之一，这个在论文中有说

![image-20250319202958076](images/image-20250319202958076.png) 

编码器的输入 是 `I times d`  $I$ 表示 输入序列长度，在这里例子就是 36，$d$ 是特征数，这里的特征数，都去掉了时间戳，也就是 7

解码器的输入是 `二分之 I + O`，`二分之 I `表示 输入序列长度的一半，`O` 表示预测步长，也就是输出序列的长度

batch x mark，batch y mark 就是处理的时间戳特征了，包含一天的第几个小时，一个月的第几天，一周的第几天，一个月的第几天，就是我们之前讲过的 SegRNN，这里处理还涉及了 归一化 和中心化，不再重复啦。

---

**好了，接下来进入 预测部分，==步进==，也就是 predict 函数** 

首先，构造完整的解码器输入，具体的操作是，切片 batch y 中的预测步长，填充 0，并与 之前的 label length 进行拼接。也就是这两行代码

```python
# decoder input 
# 创建解码器输入的零张量部分，用于预测未来时间步
# batch_y[B, label_len+pred_len, D] -> 切片 -> [B, pred_len, D] -> 创建相同形状全零张量 -> dec_inp[B, pred_len, D]
dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()

# 将历史数据(标签序列)与零张量连接，形成完整的解码器输入，并移动到指定设备
# [B, label_len, D] + [B, pred_len, D] -> torch.cat沿维度1拼接 -> [B, label_len+pred_len, D] -> to(device) -> 在GPU上的dec_inp
dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
```

构造的完整解码器的输入，形状还是 32,42,7。

（这里的代码并不是那么重要，所以就不粘贴了，占地方）接下来是一个内部方法 run model，类似 forward，但因为不是一个具体的模型，所以就叫 run model了，类内调用了这个函数，才会执行，这里没有调用，进入下一步，判断是否采用了自动精度训练，我也不明白，大概是模型加速把，总之是 false，执行 else。

```python
else:
    # 使用普通精度执行模型计算
    # _run_model() -> outputs[B, label_len+pred_len, D]
    outputs = _run_model()
```

调用的内部方法 `_run_model()`，步进，进入到 run model 内部。

![image-20250319211512141](images/image-20250319211512141.png)

首先，这里的 self.model 是 `Exp_Basic`中的 `build_model` 定义来的，而且`exp_main` ， `Basic` 的子类 重写了 父类的方法，并通过字典，键是字符串，值的类，索引进行类的初始化，这个也是 SegRNN 中介绍过的。总之，这里的 `self.model` 是 `Autoformer` 

![image-20250319211736243](images/image-20250319211736243.png)

**点击步进，进入 Autoformer 的 forward 中。一个 batch 中样本的处理** 

----

首先，这里Autoformer  forward 接收的参数：

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask=None, 
            dec_self_mask=None, 
            dec_enc_mask=None):
```

必须传入的参数 是  `x_enc, x_mark_enc, x_dec, x_mark_dec` 我们这里就是 `batch x，batch y，batch x mark，batch y mark`，且形状分别是 `[32,36,7]、[32,42,7]、[32,36,4]、[32,42,4]`

可选参数是 Transformer 中的 3 个 mask，默认是 None。解释一下 Transformer 中的三个 mask 分别是什么：

> 三个mask机制，分别指的是
>
> - 第一个 编码端输⼊ 由于padding字符的mask，为了⼀个batchsize中，所有长度不相同的样本，能构成⼀个矩阵，所以有pad字符，但是在后⾯进⾏inputencoder的⾃注意⼒计算时，pad字符不能影响计算结果，所以需要mask；
> - 第⼆个mask是解码端的mask，这个mask是涉及到因果的mask，因为Transformer是⼀个⾃回归模型，在进⾏运算时，为了并⾏计算，我们是把inputs和outputs⼀起喂给模型的，inputs直接给模型没事，但是outputs在得到最后的输出时，不能借助未来信息，只能是当前时刻及其之前时刻的输出，所以需要⼀个mask机制，这个mask是⼀个上三角矩阵，保证在预测当前输出时，不会借助未来信息。
> - 第三个mask，是编码器和解码器的交互注意⼒，编码器的输出作为key和value，解码器的输出作为query，因为⽬标序列 每个样本的长度是不⼀样的，同时原序列的样本长度也是不⼀样的，⽽且⼀对之间 长度也是不⼀样的，所以需要⼀个mask 将原序列中某个单词某个位置 跟 ⽬标序列中 某个位置 如果它们之间 有⼀个pad的话 说明是⽆效字符，得到这样的掩码矩阵。
>
> 编码器以及 编码器和解码器的 mask 是为了保证长度的对齐，解码器的 mask 是为了在预测时 避免看到未来的信息

回到 Autoformer 这里，看这个模型是怎么处理，输入数据和输出数据，以及模型的创新是怎么实现的。

首先，看到下面这几行代码。

![image-20250319214129515](images/image-20250319214129515.png)

这几行代码的目的是为了解码器的输入的初始化，编码器阶段是用不到。

---

**看论文 输入序列的趋势序列和季节趋势是怎么提取的。** 

本文将时间序列分解为 趋势序列和季节向量

![image-20250319214344038](images/image-20250319214344038.png)

趋势向量反映了数据的长期变化趋势和季节趋势。并且论文中提到 对于未来序列进行分解是不现实的，因为未来的所有序列都是不知道的。因此，为了解决这个问题，原文提出了 序列分解模块，思想是 从预测的中间隐藏变量中 逐步提取 长期稳定的趋势 。

具体的做法，使用移动平均来平滑周期性波动来突出长期趋势。

文中也给出了公式：

![image-20250319220946468](images/image-20250319220946468.png)

公式的解释：对于长度 为 L 的输入序列 X ，形状是 L×d，使用平均池化进行移动平均，并且使用填充操作保持序列长度不变。后面用一个 SeriesDecomp(X)来表示 上面的过程，简化一下记号。

**论文中的模型结构图也有画出这部分**

![image-20250319221443646](images/image-20250319221443646.png)

首先 箭头指的地方时 直观地显示了 输入序列 趋势序列 和 季节序列是怎么来的。输入序列 的 趋势序列 是对 输入序列 去均值；季节信息，也就是周期波动信息是 输入序列 - 均值 ，这个周期波动信息 是围绕 0 进行波动的。基于对输入序列的分解的认识，对于解码器 趋势序列 和 季节序列的 初始化也是很有道理的。

图片的下半部分，是解码器的输入，显示了 预测序列 趋势序列和季节序列的初始化，其中趋势序列使用输入序列的均值进行初始化，季节波动信息用 0 来初始化

---

**接下来，看代码中，对预测序列 的 趋势序列 和 季节序列的提取。**

首先 有 历史数据 x_enc [B, L, D]的，预测和标签数据 x_dec [B, L+P, D]，接着进行时间序列分解 将历史序列分解为季节性和趋势两个成分

```python
seasonal_init, trend_init = self.decomp(x_enc)
```

得到 趋势初始值：历史序列均值，季节性初始值：全零张量

基于 输入序列的 序列分解结果，构造 解码器的输入，具体来说：

- 输出序列 趋势输入= 历史趋势末尾 + 趋势初始值

- 输出序列 季节性输入 = 历史季节性末尾 + 季节性初始值(零)

也就是源码中的这几行：

```python
mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device) 
trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
```

 这个模型的结构简单来说是 利用 **编码器**处理历史数据，**解码器**利用编码器输出和组装的初始输入生成预测，就是一个很标准的 Transformer 处理数据的架构。我们得到的最终输出是 趋势和季节性预测相加，因为有 label length，所以对于输出 是 提取末尾 pred_len 长度作为最终预测结果

Autoformer 的核心思想就是 将时间序列分解为不同频率成分并分别建模，再组合生成最终预测。

**先有个大体的印象，后面看到代码 详细的讲解。**

---

在进行后面的Encoder 和 Decoder之前，**先看 趋势项 和 季节项 的具体实现方法。** 有点复杂，但是一步步来。

▶️ 首先是调用 的   self.decomp

```python
seasonal_init, trend_init = self.decomp(x_enc)
```

▶️ 而 self.decomp 又是 初始化 series_decomp 类

```python
self.decomp = series_decomp(kernel_size)
```

▶️ 看到 series_decomp 类的定义

```python
class series_decomp(nn.Module):
```

🟢 类的定义

```python
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
```

▶️ 类内 调用 `moving_avg` 

![image-20250317202431440](images/image-20250317202431440.png)

▶️ 看到 `moving_avg` 类的定义

```python
class moving_avg(nn.Module):
```

🟢 `moving_avg` 定义

```python
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series

        # 提取第一个时间步并重复，用于前端填充
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
```

总结：就是 3 次调用：

```python
seasonal_init, trend_init = self.decomp(x_enc)

self.decomp = series_decomp(kernel_size)

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        
    def forward(self, x):
        moving_mean = self.moving_avg(x)

class moving_avg(nn.Module):
     def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) 
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x	
```

用一张图表示 Autoformer 序列分解的的过程，这个分解过程将原始序列 x_enc 分解为两个相同形状 [B,L,D] 的张量：趋势成分和季节性成分：

```
                    输入: x_enc [B, L, D]
                          |
                          v
            +---------------------------+
            | Model.forward()           |
            | 调用: self.decomp(x_enc)  |
            +---------------------------+
                          |
                          v
            +---------------------------+
            | series_decomp(kernel_size)|
            | self.decomp实例           |
            +---------------------------+
                          |
                          v
            +---------------------------+
            | series_decomp.forward(x)  |
            | 1. 调用移动平均计算趋势   |
            | 2. 原序列减去趋势得到季节性|
            +---------------------------+
                          |
                  +-------+-------+
                  |               |
                  v               v
    +---------------------------+  +---------------------------+
    | moving_avg.forward(x)     |  | 季节性计算                |
    | 步骤:                     |  | res = x - moving_mean     |
    | 1.前后填充序列           |  |                           |
    | 2.应用平均池化           |  |                           |
    | 3.返回趋势分量           |  |                           |
    +---------------------------+  +---------------------------+
                  |               |
                  v               v
             趋势分量        季节性分量
          trend_init [B,L,D]  seasonal_init [B,L,D]
                  |               |
                  +       +       +
                          |
                          v
                返回到Model.forward()
                进行后续处理
```

讲图 逐字稿：

（1）**Model.forward()** 调用 self.decomp(x_enc)进行序列分解

（2）**series_decomp.forward(x)**

> 包含两个主要步骤:
>
> - 调用 self.moving_avg(x)计算移动平均，得到趋势分量
> - 计算原序列与趋势分量的差值，得到季节性分量

（3）**moving_avg.forward(x)**

> 执行移动平均计算:
>
> - 通过重复首尾元素进行序列填充
>
> ```python
> front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) 
> end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
>  x = torch.cat([front, x, end], dim=1)
> ```
>
> - 应用一维平均池化操作
>
> ```
> x = self.avg(x.permute(0, 2, 1))
> ```
>
> **说明 为什么填充，是为了 保证序列在平均池化后 长度不变**
>
> - 返回平滑后的趋势分量
>
> 这部分的形状变化： 
>
> ![image-20250319224010427](images/image-20250319224010427.png) 

现在开始 返回 **moving_avg.forward(x)** 是利用 1D 平均池化 得到 趋势序列，将结果返回给 **series_decomp** ，也就是这句代码 `moving_mean = self.moving_avg(x)`，得到趋势序列以后，永远序列减趋势序列 `res = x - moving_mean` ，得到季节分量，也就是周期性信息。具体的代码：

![image-20250319224315222](images/image-20250319224315222.png)

最终 将结果 返回给 Autoformer forward 中的 seasonal_init, trend_init

![image-20250319224700908](images/image-20250319224700908.png)

并且 用这两个 init 初始化 解码器的输入。

这里得注意一下，对于 标签序列，也就是 输入序列的趋势信息的提取用的是 1D平均池化，而对预测 predict length 的趋势信息初始化 就直接用的 输入序列的均值

```python
mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
```

周期性趋势也是，label length 的季节趋势是 残差，也就是 原始序列 减去 趋势序列，而 predict length 的 季节趋势就是直接初始化为 0 了。

```python
zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device) 
```

这里是 小小的区别，小小的注意。

好了 这部分，序列分解说完了，代码讲了，原文讲了，公式对应上了，图也说了。原文 `Series decomposition block`  就过啦

![image-20250319225341745](images/image-20250319225341745.png)





序列分解 over

---

下面开始 模型的输入，先从论文开始讲解：

![image-20250319225433558](images/image-20250319225433558.png) 



模型的输入部分，模型的输入包括编码器的输入和解码器的输入。具体来说，

编码器的输入是过去 $I$ 个时间步，文中给出的符号表示 $\mathcal{X}^{I \times d}$ ，$I$ 表示时间步长，$d$ 表示每个时间步的特征数。

解码器的输入包括了 季节性序列 和 趋势性序列，具体的符号表示分别是 $\mathcal{X}_{des}$  和 $\mathcal{X}_{det}$    形状是一样的：$(\frac{I}{2}+O)$  、$\frac{I}{2}$ 是 label length 的长度，取的是原始输入序列长度的 一半。O 是 预测步长 predict length。d 同样是每个时间步的特征数。接下来，我们来看公式是怎么表示的：

![image-20250320085551155](images/image-20250320085551155.png)

$\mathcal{X}_{ens}、\mathcal{X}_{ent}$  分别表示 从 原始 输入序列 $\mathcal{X}_{en}$ 分解出的季节成分和趋势成分，截取出后半部分 $\frac{I}{2}:I$ 作为 label length，与长度为 predict  length 的时间步进行拼接，用 0 填充的长度为 predict length的向量记作 $\mathcal{X}_0$ ，用输入时间序列时间步均值填充的长度为 predict length 的向量记作 $\mathcal{X}_{mean}$

然后，$\mathcal{X}_{ens}$ 与 $\mathcal{X}_0$ 进行 concat 得到 解码器季节成分的初始值  $\mathcal{X}_{des}$

 对应着的 $\mathcal{X}_{ent}$ 与 $\mathcal{X}_{mean}$ concat 得到解码器趋势成分的初始值 $\mathcal{X}_{mean}$

**再强调一下，这里所涉及的向量的记号和形状：** 

- 编码器的输入是 过去 $I$ 个时间步，表示 $\mathcal{X}^{I \times d}$ ，$I$ 表示时间步长，$d$ 表示每个时间步的特征数。
- 解码器季节成分的输入是 $\mathcal{X}_{des} ^{(\frac{I}{2}+O)\times d}$ 、解码器趋势成分的输入是 $\mathcal{X}_{det} ^{(\frac{I}{2}+O)\times d}$ 
- 涉及到的中间变量，$\mathcal{X}^{\frac{I}{2} \times d}_{ens}$ ，$\mathcal{X}^{\frac{I}{2} \times d}_{ent}$ 可以理解为标签序列的季节成分和趋势成分，就是从输入序列分解的季节成分和趋势成分中截取的后半段。
- 预测序列季节成分的初始值是 $\mathcal{X}_0 ^{O \times d}$ ，趋势成分初始值是 $\mathcal{X}^{O \times d} _{Mean}$

也就是论文中模型结构图的：

![image-20250320091412914](images/image-20250320091412914.png)

具体到代码，就是 autoformer forward的前 5 行，其中 self.decomp是我们刚刚仔细讲过的 序列分解模块 Series decomposition block：

![image-20250320091921116](images/image-20250320091921116.png)

这部分代码比较好理解，就这样，以上部分完成了对原文 model inputs 部分的讲解，代码，论文，图，公式都讲了。

![image-20250320092158923](images/image-20250320092158923.png)

![image-20250320092214513](images/image-20250320092214513.png)

**接下来进入论文的 Encoder 部分**  

会同样按照，论文、图、公式、代码一一对应的逻辑进行讲解

**首先，Autoformer 遵循原始 Transformer 的结构，** 

![image-20250320092445230](images/image-20250320092445230.png)

编码器，解码器，编码器接收的 input 是 word embedding + positional embedding，然后通过自注意力机制和前馈神经网络。解码器接收的 输入是 output，预测部分，同样是 word embedding+positional embedding，然后分别经过解码器输入的 自注意力机制，以及和编码器输出 的 交叉注意力，最后经过 全连接层，得到最终的输出。

首先强调一下关于Transformer 为什么是注意力机制和全连接层的设计？

> 首先，Transformer 在 NLP中接收的数据格式 是 [B,L,D]，batch size，一个 batch 中有多少个句子，一个句子中有几个词 L，每个词的嵌入D，也就是每个词用长度为多少的向量表示
>
> 最直观的讲解，就是 注意力机制进行 L 层面的交互，前馈神经网络进行 D 层面的交互。
>
> **L 层面也就是注意到了 词与词之间的交互，D 层面就是词与词之间特征的交互** 
>
> > 在L层面（单词层面）进行交互，计算每个单词对其他单词的注意力权重，捕捉词与词之间的关系；
> >
> > 在D层面（即单词嵌入的特征层面）进行交互，对每个单词的嵌入向量进行非线性变换，捕捉词与词之间的特征交互 
>
> **对应到时间序列中**
>
> 1️⃣ 标准 ==输入== 格式也是 BLD，具体的解释： 
>
> > B = 32 (批量大小，32个时间序列样本)
> > L = 36 (每个样本有36个时间步，如过去36天的数据)
> > D = 7 (每个时间步有7个特征，如对于股票可能包括开盘价、收盘价、最高价、最低价、交易量等)
>
> 2️⃣ ==处理==   注意力机制
>
> 编码器中，注意力在所有36个时间步之间建立连接
> 解码器中，注意力既在预测序列内部建立连接，也与编码器输出建立连接
>
> 时间步之间的建模 可以 发现股票价格每周五可能下跌，或者每月初可能上涨的模式
>
> 3️⃣ ==处理==  前馈全连接层
>
> 处理每个时间步内7个特征之间的关系
>
> 例如，交易量与价格变动的关系，或开盘价与收盘价的关系

诶，说起这个，关于用现实例子理解这些模型，

**首先，卷积是什么意思？** 

假如我们要认识一个人A，B 是 A 的直接朋友，形成了B 对 A 的第一次认识，B 就相当于卷积核了，那直接认识 A的肯定不止一个人，还有B1，B2，B3...等，每个人对形成了对 A 的第一次认识，父母认识 A更关注生活层面，学校中直接认识的 A 更关于为人处事部分，工作中直接认识的 A 更关于 A 的生产性。这里直接认识 A 的B1，B2，B3...就是每一层中 卷积核的个数。除了直接认识 A 的，还有通过直接认识 A 的人B 认识 A，这波人叫 C，那还有通过 C 认识 A 的，那 C 又认识 D，D 又通过 C 认识 A。除了别人认识 A，A 自己也有对自己的认识。

**Transformer是什么意思？**

除了刚刚说的 注意力机制和前馈全连接层的理解，还有 Encoder 、Decoder 、多头注意力机制的理解。

- [x] Encoder&Decoder 的交互怎么理解？

首先，整体上的这个图：

![image-20250320095403477](images/image-20250320095403477.png)

编码器相当于甲方，解码器相当于乙方，甲方有需求，自己公司内部一级一级沟通，从最开始的想法最终形成方法交给最后一个人，这个人去和乙公司沟通，乙公司又有很多个部分，每个部分分别完成甲公司提出的方案的一部分，这一个过程中需要不断的与甲公司手拿最终方案的人不断沟通，最终乙公司完成方案。

- [x] 多头注意力机制怎么理解

对于 BLD 的序列，首先明白的是，那个维度分多头了，是 D 维度分成 num head维度和 head dim，其中 num head × head dim = embedding dim（D），相当于什么意思，一个人学知识（B =1），L 是要学的几本书，D 是每本书有几个章节，一般是一个老师教我们学一整本书，但多头注意力机制的意思是，一本书的几个章节，分开，比如第一个老师教第一章和第二章，第二个老师教第三章和第四章，最后两张第三个老师教，这样学习的时候，同样是一个学期，一个老师只需要关注两章的内容，对于课程节奏的把握知识理解的更透彻，效果会比一个老师教一整本书的内容要好一些。

B=3，就是班里的 3 个人，每个人这学期都要上这几本课，同样的 LD。

> 最后一个 linear 层，应该是为了还原原始维度的。

**好了，扩展的远了，回到论文中Encoder 部分** 

![image-20250320092249326](images/image-20250320092249326.png)

原文中说，首先编码器更专注季节部分的建模，编码器的输出包含过去的季节性信息，并将作为交叉信息帮助解码器细化预测结果，假设有 N 个编码器，则第 i 层编码器的总体方程可以表示为 $\mathcal{X}_{en}^l = Encoder(\mathcal{X}_{en}^{l-1})$ ，就是说 第 $l$ 层编码器接收 第 $l-1$ 层编码器的输出作为输入，具体的细节是原文的公式(3)

**下面对 公式 3 进行讲解**

首先，等号左边，下划线表示忽略掉季节成分，只关注季节成分。

$\mathcal{X}_{en}^l = \mathcal{S}_{en}^{l,2},l \in {1,...,N}$  表示 第 $l$ 层编码器的输出。

- 初始值，也就是编码器的输入是 $\mathcal{X}_{en}^0$ 是 输入时间序列的 $\mathcal{X}_{en}$ 的 word embedding

- [ ]  $\mathcal{S}_{en}^{l,i},i \in {1,2}$ 表示 第 $l$层中 第 i 个序列分解模块之后的季节性成分，然后公式中的 Auto-correlation 后面再说，这是本文的一个创新点。（ps，后面要重点看这个是什么意思。）

> （我最开始看见这里的疑问，不用讲，忽略掉即可）先看公式等号的左边， $\mathcal{S}_{en}^{l,1}$ 首先，下标 $en$ 就是表示 编码器，$l$ 表示第几个编码器，那这个 $1$是什么意思？

**原文和公式说了，接下来来看代码，Encoder 是怎么实现的。**

==首先，构造 Encoder 的输入== ，编码器嵌入。

![image-20250320143146767](images/image-20250320143146767.png)

具体怎么做的看autoformer 的 init 部分：

![image-20250320143228968](images/image-20250320143228968.png)

看到这边调用的 `DataEmbedding_wo_pos` 这个类，其中具体地 valueEmbedding 和TemporaryEmbedding 又分别在 init 中显示调用了 `TokenEmbedding` 类和 `TemporalEmbedding` 类

![image-20250320143256379](images/image-20250320143256379.png)

嵌入部分的调用关系用流程图来表示：

```mermaid
classDiagram
    class DataEmbedding_wo_pos {
        +TokenEmbedding value_embedding
        +PositionalEmbedding position_embedding
        +TemporalEmbedding temporal_embedding
        +Dropout dropout
        +forward(x, x_mark)
    }
    
    class TokenEmbedding {
        +Conv1d tokenConv
        +forward(x)
    }
    
    class PositionalEmbedding {
        +Tensor pe
        +forward(x)
    }
    
    class TemporalEmbedding {
        +Embedding minute_embed
        +Embedding hour_embed
        +Embedding weekday_embed
        +Embedding day_embed
        +Embedding month_embed
        +forward(x)
    }
    
    class TimeFeatureEmbedding {
        +Linear embed
        +forward(x)
    }
    
    DataEmbedding_wo_pos --> TokenEmbedding
    DataEmbedding_wo_pos --> PositionalEmbedding
    DataEmbedding_wo_pos --> TemporalEmbedding
    TemporalEmbedding --> TimeFeatureEmbedding
```

首先，跟大家说这个图怎么画，首先在调试的过程中，看到调用相关的代码，就粘贴给 gpt，然后让 gpt 画。这个图就是 gpt 给我画的，它用的 mermaid ，生成代码，然后我粘贴到我的 markdown 文档中，我用的 markdown 编辑器是 Typora，可以解析 mermaid，用在线mermaid 也可以显示出图。直接搜 在线 mermaid。或者跟 gpt 说，用简单的流程图画，不用 mermaid，都能帮你把自己的代码理清楚。

mermaid 画出的类调用图，一个类用三行表示，第一行 类名、第二行，init 部分的定义、第三行类中方法的定义

**好了，现在开始讲图，** 

可以看到 `DataEmbedding_wo_pos` 类 的 init 分别调用了 `TokenEmbedding`类、`PositionalEmbedding`类和 `TemporalEmbedding`类，同时还定义了一个 dropout 层。

🔵 调用 `tokenEmbedding`类，init 部分是使用一个 `nn.Conv1d` 初始化了一个卷积层，传给 `self.tokenConv` ，后面在 这个类中的 forward 方法中用。

![image-20250320144456851](images/image-20250320144456851.png)

通俗点说，这里的 tokenEmbedding 就是通过一个1D 卷积实现的，具体的形状变化注释中也给出了。

> 怎么生成注释？
>
> 首先把代码粘给 gpt，然后，跟它说：`为每行代码 添加 两行注释，一行说明这行代码的目的，一行说明 形状的变化和操作 形状->操作->形状的格式，操作的格式类似 DecoderLayer.forward 显示出调用的什么类名.方法`

🔵 接下来看 位置编码 Positional Embedding，由于这里没有用，就不说了。

🔵 最后，时间戳编码，

![image-20250320145727249](images/image-20250320145727249.png)

注意这里的时间戳编码是有一个判断的，经过调试，我们这里调用的是 `TimeFeatureEmbedding` 类。

也就是说什么意思，这个图画的有问题，不过意思也是对的，就不深究了。

接下来，我们跳到 `TimeFeatureEmbedding` 这个类的定义。

![image-20250320150012753](images/image-20250320150012753.png) 

就是通过一个线性层，将 时间戳特征嵌入到指定维度。

首先，嵌入到指定维度是因为高维向量表示特征更精细。

其次，我们这里使用的是疾病数据集，是小时的，所以维度 4，表示的的是，小时-天，天-周，天-月，天-年。这一部分也说过很多次了，再说一次，加深印象。

具体来说输入的 `x_mark.shape=32,36,4 → nn.Linear → 32,36,512`

接下来，总结一下这里的嵌入。首先 本文用到的所有嵌入都定义在了  `Embed.py`文件中

![image-20250320150552474](images/image-20250320150552474.png)

而这个文件中，又定义了所有的嵌入类，又有 8 个。

> 题外话，这个怎么看，是 vscode 的大纲视图，找出来，就能看到了
>
> ![image-20250320150811010](images/image-20250320150811010.png)
>
> 大纲视图中，立方体表示定义的函数，小树杈的东西是类，类中有小立方体，是类中定义的函数，类中定义的函数，也就是小立方体中，折叠的部分是 使用这个函数或者类所需要的初始化参数。方括号+小立方体包括的部分是 类中调用的类的对象名，比如这里：
>
> ![image-20250320151133019](images/image-20250320151133019.png)
>
> 以这个 TemporalEmbedding 类为例， 这个TemporalEmbedding 类中有两个方法方法，分别是 init 和 forward。
>
> init 折叠的部分是 初始化这个类所需要的参数， forward 折叠的部分是调用这个时所需要的参数，其中 init 部分还实例化了 5 个对象，对象名分别是 mintue_embed、hour_embed、weekday_embed、day_embed、month_embed，但是这里具体实例化的哪个类。这里是没有显示的，得点进去自己看，可以看到这个对象其实都是实例化的Embed 这个类，很明显是一个自定义的，想看还得步进看具体实例化的哪个类。

以上完成了 Encoder Input 的 Embedding 部分，分别进行了 token Embedding 和 TemporaryEmbedding来对历史时间步特征进行嵌入和时间特征进行嵌入。

汇总这里的维度形状变化：

```python
# x [B, L, D] → permute → [B, D, L] → 卷积 → [B, d_model, L] → transpose → [B, L, d_model]
# x_mark [B, L, d_inp] → 线性层变换(时间特征整体映射) → [B, L, d_model]
# [B, L, d_model] + [B, L, d_model] → [B, L, d_model]
```

**接下来想给大家说的是， 1D 卷积怎么进行的 tokenEmbedding：** 

小小的点，小小的注意。

接收的标准输入是 BLD

- 首先进行的是 permute，将想要嵌入的维度`D` 移到中间，然后进行 1D 卷积，嵌入到 `d_model`  （ `Embedding dim`），对应到 1D 卷积中，就是输入通道是 D，输出通道是 `d_model`

![image-20250320152707837](images/image-20250320152707837.png)

- 为什么这么做？因为卷积最开始主要用于图像，图像的标准格式是 BCHW，图像中的 HW 就表示图像的特征，只不过是用 2D的矩阵 表示的，而且这个 2D 矩阵保存了位置信息，不能随意展平。那此时，C 也就可以理解为每个像素的特征数。比如每个像素用彩色的 RGB 三个元素表示。
- 所以我们这里的时间序列中的 1D 卷积，也仿照图像中卷积的定义，每个时间步的特征数放到中间，表示输入通道数，然后将每个时间步的特征，映射到输出维度大小，这里表示为 `Embedding dim`，也就是 `d_model`。

用一张图来表示，(这里其实很像 SegRNN 的视角转换)：

![image-20250320154515642](images/image-20250320154515642.png)

而 时间戳特征的 nn.Linear就是直接对最后一个维度进行嵌入了



