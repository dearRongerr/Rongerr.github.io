# SegRNN实验过程

现在需要弄清楚，怎么复现出论文的结果。

论文中表 2 给出了所有用到的模型和数据集。

[https://github.com/lss-1138/SegRNN](https://github.com/lss-1138/SegRNN)

源码仓库给了复现论文结果的脚本命令，我已经用了 AutoDL 跑了这个脚本，按理说是已经复现了一部分结果。

我现在对 Electricity 数据集进行了数据集描述。

现在思考怎么把自己已经做得工作融入到 SegRNN 中，并以 SegRNN 为基础，进一步的改进模型。

==现在继续看怎么看已经复现出的结果和论文中的结果进行对比。==

- checkpoints保存的是模型的 pth 参数文件

数据集数量：

1. Electricity
2. ETT h1
3. ETT h2
4. ETT m1
5. ETT m2
6. illness
7. traffic
8. weather

==汇总每个数据集做了几个实验== 

- 4 个实验||Electricity 数据集：回望窗口 96，分别预测 96,192,336,720

```
seq_len=96
for pred_len in 96 192 336 720
```

- 对应着有 4 个 logs 文件，存了结果

==问题：这个 logs 文件的结果怎么阅读？==

分析，去看 logs 文件都存了什么内容？

- logs 文件的起始点

> 分析：`logs/LongForcasting` 文件夹下
>
> > `logs/LongForecasting/SegRNN_Electricity_720_96.log`
> >
> > `logs/LongForecasting/SegRNN_Electricity_720_192.log`
> >
> > `logs/LongForecasting/SegRNN_Electricity_720_336.log`
> >
> > `logs/LongForecasting/SegRNN_Electricity_720_720.log`
>
> 保存了这个脚本的数据文件：
>
> > `/root/SegRNN/scripts/SegRNN/electricity.sh`
>
> **总结**：这个脚本跑了回溯窗口 96，分别进行 未来 96 步长，192 步长，336 步长，720 步长的预测

现在就先分析 `/root/SegRNN/logs/LongForecasting/SegRNN_Electricity_720_96.log`这个 log 中记录的内容

第一部分，不用管，是参数设置 namespace

<details>
<summary>Electricity Namespace</summary>
<p>
Args in experiment:
Namespace(activation='gelu', affine=0, batch_size=16, c_out=7, channel_id=1, checkpoints='./checkpoints/', d_ff=2048, d_layers=1, d_model=512, data='custom', data_path='electricity.csv', dec_in=7, dec_way='pmf', decomposition=0, des='test', devices='0,1', distil=True, do_predict=False, dropout=0.1, e_layers=2, embed='timeF', embed_type=0, enc_in=321, factor=1, fc_dropout=0.05, features='M', freq='h', gpu=0, head_dropout=0.0, individual=0, is_training=1, itr=1, kernel_size=25, label_len=0, learning_rate=0.0005, loss='mae', lradj='type3', model='SegRNN', model_id='Electricity_720_96', moving_avg=25, n_heads=8, num_workers=10, output_attention=False, padding_patch='end', patch_len=16, patience=10, pct_start=0.3, pred_len=96, random_seed=2024, revin=0, rnn_type='gru', root_path='./dataset/', seg_len=48, seq_len=720, stride=8, subtract_last=0, target='OT', test_flop=False, train_epochs=30, use_amp=False, use_gpu=True, use_multi_gpu=False, win_len=48)
Use GPU: cuda:0
</p>
</details>

第二部分，