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

```
Electricity_720_96_SegRNN_custom_ftM_sl720_pl96_dm512_dr0.1_rtgru_dwpmf_sl48_mae_test_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 17597
val 2537
test 5165
```

区分 iteration和 epoch

epoch 设置为 30  `train_epochs=30`

<details>
<summary>namespace 参数解释</summary>
<p>
```
    Electricity_720_96_SegRNN_custom_ftM_sl720_pl96_dm512_dr0.1_rtgru_dwpmf_sl48_mae_test_0：这是训练任务的标识符，包含了模型和训练设置的关键参数：
Electricity：数据集名称。
720：回溯长度（seq_len）。
96：预测长度（pred_len）。
SegRNN_custom：模型名称。
ftM：特征类型（M 表示多变量）。
sl720：序列长度（seq_len）。
pl96：预测长度（pred_len）。
dm512：隐藏层维度（d_model）。
dr0.1：丢弃率（dropout）。
rtgru：RNN 类型（GRU）。
dwpmf：解码方式（Parallel Multi-step Forecasting）。
sl48：段长度（seg_len）。
mae_test_0：使用均方误差（MAE）作为损失函数，测试集编号为 0。
```
</p>
</details>

训练集样本：17597，`batch_size=16`

所以，每个 epoch 的 iters = $17597/16=1,099.81 $

以上是第一个 epoch=1 的输出

```
	iters: 100, epoch: 1 | loss: 0.6690567
	speed: 0.0366s/iter; left time: 1203.1702s
	iters: 200, epoch: 1 | loss: 0.4867136
	speed: 0.0277s/iter; left time: 907.4625s
	iters: 300, epoch: 1 | loss: 0.4125512
	speed: 0.0258s/iter; left time: 844.1005s
	iters: 400, epoch: 1 | loss: 0.3804378
	speed: 0.0270s/iter; left time: 878.8473s
	iters: 500, epoch: 1 | loss: 0.3569579
	speed: 0.0235s/iter; left time: 762.6345s
	iters: 600, epoch: 1 | loss: 0.3551046
	speed: 0.0255s/iter; left time: 823.9151s
	iters: 700, epoch: 1 | loss: 0.3524055
	speed: 0.0240s/iter; left time: 774.8318s
	iters: 800, epoch: 1 | loss: 0.3217510
	speed: 0.0227s/iter; left time: 731.5172s
	iters: 900, epoch: 1 | loss: 0.3337467
	speed: 0.0229s/iter; left time: 733.8034s
	iters: 1000, epoch: 1 | loss: 0.3291512
	speed: 0.0232s/iter; left time: 741.6312s
Epoch: 1 cost time: 28.068471431732178
Epoch: 1, Steps: 1099 | Train Loss: 0.4208171 Vali Loss: 0.2863213 Test Loss: 0.3096032
```

上面已经算了，根据训练样本数和 batch 的大小，一次 epoch 需要 1000 次 iter

模型100 次 iter 记录一次损失，每次迭代速度以及剩余的时间。

```
	iters: 100, epoch: 1 | loss: 0.6690567
	speed: 0.0366s/iter; left time: 1203.1702s
```

- iters：当前迭代次数。
- epoch：当前训练轮数。
- loss：当前迭代的损失值。
- speed：每次迭代的平均耗时（单位：秒/迭代）。
- left time：预计剩余训练时间（单位：秒）。

训练完以后，输出训练日志信息：

```
Epoch: 1 cost time: 28.068471431732178
Epoch: 1, Steps: 1099 | Train Loss: 0.4208171 Vali Loss: 0.2863213 Test Loss: 0.3096032
```

- **Steps: 1099： **    （是的，这个是准确的迭代步数。

表示在第一个 epoch 中，模型完成了 1099 次迭代（或步骤）。这通常是因为训练数据集被分成了 1099 个批次（batches）。

- **训练损失（Train Loss）** 

值：0.4208171：训练损失较高，表明模型在训练数据上可能还有改进空间

- **验证损失（Vali Loss）**

值：0.2863213：验证损失低于训练损失，这可能表明模型在验证集上表现良好，没有过拟合。

- **测试损失（Test Loss）**

值：0.3096032：测试损失略高于验证损失，这是正常现象，因为测试集通常包含更多未见过的数据。测试损失的值表明模型在独立测试数据上的表现。

- **时间消耗** 28.07 秒/epoch：完成一个 epoch 的时间相对较短，这表明模型训练效率较高。

还有迭代最后一部分的日志信息：

```
Validation loss decreased (inf --> 0.286321).  Saving model ...
Updating learning rate to 0.0005
	iters: 100, epoch: 2 | loss: 0.2965542
	speed: 0.1480s/iter; left time: 4702.0987s
	iters: 200, epoch: 2 | loss: 0.2877236
	speed: 0.0264s/iter; left time: 836.8543s
	iters: 300, epoch: 2 | loss: 0.2788590
	speed: 0.0271s/iter; left time: 854.0580s
	iters: 400, epoch: 2 | loss: 0.2706560
	speed: 0.0276s/iter; left time: 868.6653s
	iters: 500, epoch: 2 | loss: 0.2822747
	speed: 0.0263s/iter; left time: 824.1755s
	iters: 600, epoch: 2 | loss: 0.2616775
	speed: 0.0269s/iter; left time: 840.6318s
	iters: 700, epoch: 2 | loss: 0.2763505
	speed: 0.0262s/iter; left time: 816.8357s
	iters: 800, epoch: 2 | loss: 0.2719800
	speed: 0.0259s/iter; left time: 804.4196s
	iters: 900, epoch: 2 | loss: 0.2634850
	speed: 0.0263s/iter; left time: 813.3547s
	iters: 1000, epoch: 2 | loss: 0.2671717
	speed: 0.0265s/iter; left time: 817.9329s
```

重点看这个：

```
Validation loss decreased (inf --> 0.286321).  Saving model ...
Updating learning rate to 0.0005
```

Validation loss decreased：表示验证损失值下降了。

inf --> 0.286321：表示在上一个 epoch 中，验证损失值为无穷大（inf），现在下降到了 0.286321。

Saving model：表示由于验证损失值下降，模型被保存。

Updating learning rate：表示学习率被更新了。

to 0.0005：表示新的学习率是 0.0005。

```
	iters: 100, epoch: 2 | loss: 0.2965542
	speed: 0.1480s/iter; left time: 4702.0987s
```

哦，不是啊，这已经是第二次迭代的训练信息了。

以第二个 epoch 为例，查看所有的日志信息：

```
Updating learning rate to 0.0005
	iters: 100, epoch: 2 | loss: 0.2965542
	speed: 0.1480s/iter; left time: 4702.0987s
	iters: 200, epoch: 2 | loss: 0.2877236
	speed: 0.0264s/iter; left time: 836.8543s
	iters: 300, epoch: 2 | loss: 0.2788590
	speed: 0.0271s/iter; left time: 854.0580s
	iters: 400, epoch: 2 | loss: 0.2706560
	speed: 0.0276s/iter; left time: 868.6653s
	iters: 500, epoch: 2 | loss: 0.2822747
	speed: 0.0263s/iter; left time: 824.1755s
	iters: 600, epoch: 2 | loss: 0.2616775
	speed: 0.0269s/iter; left time: 840.6318s
	iters: 700, epoch: 2 | loss: 0.2763505
	speed: 0.0262s/iter; left time: 816.8357s
	iters: 800, epoch: 2 | loss: 0.2719800
	speed: 0.0259s/iter; left time: 804.4196s
	iters: 900, epoch: 2 | loss: 0.2634850
	speed: 0.0263s/iter; left time: 813.3547s
	iters: 1000, epoch: 2 | loss: 0.2671717
	speed: 0.0265s/iter; left time: 817.9329s
Epoch: 2 cost time: 29.48875856399536
Epoch: 2, Steps: 1099 | Train Loss: 0.2751775 Vali Loss: 0.2280986 Test Loss: 0.2488164
Validation loss decreased (0.286321 --> 0.228099).  Saving model ...
```

==30 个 epoch，直接看到最后==

- 里面还有一个指标需要注意：`早停`   **EarlyStopping counter: 3 out of 10**


<details>
<summary>关于早停</summary>
<p>
早停机制的计数器。如果验证损失在连续 10 个 epoch 中没有改善，训练将提前停止。当前计数器值为 3，表示验证损失在最近 3 个 epoch 中没有改善。
</p>
</details>

- 看到第 30 个 epoch，`test Loss: 0.2200698`

最后的日志信息：

```
>>>>>>>testing : Electricity_720_96_SegRNN_custom_ftM_sl720_pl96_dm512_dr0.1_rtgru_dwpmf_sl48_mae_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 5165
mse:0.12902846932411194, mae:0.22005711495876312, ms/sample:2.1022345181933217
```

- [x] 这两个哪个是论文的结果？（最后的测试报告）

带着结果去论文中看

!!! success
    you can use 
   日志文件最后的输出：mse:0.1290, mae:0.2200
   论文中表格给的：Electricity 预测步长96 MSE，0.128 & MAE 0.219
   该说不说，确实是个好精巧的模型，我居然就这么成功的浮现了耶。

- 以上是 这个 `SegRNN_Electricity_720_96.log`  log 文件的解读。完成了论文表 2 的两格。

- [x] 为什么训练时也有 test loss ？

> 在每个 epoch 结束时计算模型在测试集上的损失

- checkpoints，存的应该是最好的模型参数，然后进行模型训练。

- [ ] 还有消融实验的脚本文件没有看
- [ ] 我复现了论文的几格结果？

我一个个看了 Electricity 复现出来的结果（lookback=720）

`mse:0.19964559376239777, mae:0.28901728987693787`

这个结果 720→720 居然比原论文的结果还好。

别的在小数点后 3 位比原论文结果差一些。

- ETT h1 的复现结果，偏差有点大，SegRNN 小数点后两位有偏差

- 这篇论文属于真正意义上的复现成功。结果跑得大差不差，论文从头到尾的 debug。
- 论文的环境：2块 T4 的显卡，16GB
- 看看 Autoformer能跑出几个结果。
- 如果我用学校的服务器，一个数据集一个数据集的跑，也能得到结果，大概。

你知道，什么是算力上的碾压

AutoDL 上的服务器 2 块 4090 的卡。：

```
train 17597
val 2537
test 5165
	iters: 100, epoch: 1 | loss: 0.6690567
	speed: 0.0366s/iter; left time: 1203.1702s
```

学院的服务器：

```
train 17597
val 2537
test 5165
	iters: 100, epoch: 1 | loss: 0.6692066
	speed: 0.2947s/iter; left time: 9687.1735s
```

9000 秒和 1000 秒的区别。

- 我在想，SegRNN 如果baseline 的结果都是从论文中摘过来的那为什么完整的项目文件中还有 Autoformer 的实现