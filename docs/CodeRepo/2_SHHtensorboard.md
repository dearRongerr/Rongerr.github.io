# 远程服务器&tensorboard

> [远程服务器（Linux）上TensorBoard查看训练过程](https://www.bilibili.com/video/BV1kh4y1N7Vp/?spm_id_from=333.337.search-card.all.click&vd_source=ddd7d236ab3e9b123c4086c415f4939e)

（1）SSH 连接远程服务器，进入自己的项目文件夹，打开终端

（2）screen -r 

（3）激活自己的 python 虚拟环境

（4 ） 终端运行 tensorboard 命令：

```bash
tensorboard --logdir 项目文件路径 -- --bind_all
```

终端出现 

```bash
http://worker01:6006/
```

将中间的 woker01 换成服务器的 ip 地址，端口号不变

```bash
http://192.168.58.195:6006/
```

**关于步骤 （4）的详细过程：**

（4）终端运行 tensorboard 命令：

```bash
tensorboard --logdir 项目文件路径
```

给出本地访问地址   `http://localhost:6006/` ：点击即可

此时点击会出现无法访问的情况，在终端命令行窗口给出了提示，加上 `--bind_all` 开放所有权限：

![image-20250418154127676](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504181541112.png)

终端重新输入：

```bash
tensorboard --logdir 项目文件路径 --bind_all
```

![image-20250418154415440](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504181544836.png)

此时给出的访问命令变为：`http://worker01:6006/`，依然是无法访问的状态

此时把 worker01换成自己的服务器 ip 即可，端口号 `6006` 保持不变，即正确的访问地址：

```bash
http://192.168.58.195:6006/
```

可以看到 loss、网络结构图等
