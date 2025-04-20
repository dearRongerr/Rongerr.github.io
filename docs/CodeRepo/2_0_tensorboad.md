# tensorboard可视化

> [在Pytorch中使用Tensorboard可视化训练过程](https://www.bilibili.com/video/BV1Qf4y1C7kz/?spm_id_from=333.337.search-card.all.click&vd_source=ddd7d236ab3e9b123c4086c415f4939e) 
>
> - 可视化训练过程
> - [官方 pytorch、tensorboard 可视化](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
>   * 进入PyTorch官网
>   * 左侧 `Learning PyTorch`
>   * `Visualiazing Models,DATA...` 
> - tensorboard 可以干什么？
>   * 保存网络结构图 `GRAPHS`
>   * 训练集上的损失 loss、learning_rate、验证集上的 accuracy `SCALARS`
>   * 保存权重数值分布 `HISTOGRAMS`
>   * 预测图片信息 `IMAGES`

## 关于启动

> 场景描述：让一个TensorBoard实例持续运行，监视所有实验

（1）一次性启动长期运行的TensorBoard:

```python
screen -S tensorboard_permanent
cd /home/student2023/xiehr2023/UnetTSF
tensorboard --logdir=runs --bind_all --reload_interval=5
# Ctrl+A D 分离screen
```

（2）每次只需运行您的Python脚本，无需管理TensorBoard:

```bash
python module_3.py
```

（3）一直使用相同的URL查看结果:

```bash
http://localhost:6006
```

add（1）vscode自动转发端口：

![image-20250418183116571](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504181831336.png)

add（2）可以在 python 脚本中加入启动 tensorboard

> 好处：只需运行脚本，然后在VSCode中打开端口转发，即可访问最新的TensorBoard结果。不再需要单独启动TensorBoard、记住日志路径等
