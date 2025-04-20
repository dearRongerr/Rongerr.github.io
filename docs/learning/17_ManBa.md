# Mamba

引入:

(1)首先,精读一篇论文,至少需要搞懂80%的内容,剩下20%的内容,需要在代码中寻找答案

(2)第二步,已经跑得通的代码,经典论文的代码,跑通相对来说还是比较容易的

(3)第三步,代码吃透了80%,至少需要完整的debug 过一次

- 至少知道每个参数代表什么意思
- 至少知道每个操作前后的shape

---

PART02:

读懂论文,知道每个模块的作用是什么,知道每一个模块的输入和输出是什么

接下来,GITHUB把这篇论文的代码下载到本地,根据作者提供的readme文件,安装相应的包,最终目的就是让这个代码顺利地跑起来

---

PART03:

时空注意力block,两分支结构,左分支是空间注意力,右分支是时间注意力,然后两部分通过门控融合单元直接融合

---

PART04:

代码Debug 笔记,详细记录每个操作前后特征的 shape的变化过程

注意,在GITHUB上下载的代码是光秃秃秃的,没有注释,一定要好好记笔记

## 安装

mamba 介绍,mamba是序列建模方法,用来替换这个时间注意力模块,manba 原理有点难,所幸,作者封装好了,直接调用即可.

```python
from manba_ssm import Manba
```

即可直接调用 manba

这里的安装容易出现很多问题:

- 版本和cuda对不上
- 远程下载不下来

---



- manba 主页:[https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

> 要求:Linux\NVIDIA GPU\Pytorch1.12+\CUDA 11.6+
>
> - 没有 cuda GPU 别想了

- manba原文:[https://arxiv.org/pdf/2312.00752](https://arxiv.org/pdf/2312.00752)

> - 发布日期：2023 年 12 月
> - Albert Gu 和 Tri Dao - 卡耐基梅隆大学和普林斯顿大学

-----

稳定安装的方法:

```python
# 环境: Cuda 11.8, python 3.8(ubuntu20.04), PyTorch  2.0.0

### 不稳定安装方法
# 运气好的话,一次性安装完成,运气不好,一天也安装不好, 因为是从github直接拉取资源,非常不稳定: pip install mamba-ssm --timeout=200
### 稳定安装方法
# 1. 通过此命令行查看安装的是哪个wheel文件:pip install mamba-ssm --no-cache-dir --verbose
# 2. 复制给定的.wheel链接到浏览器,直接下载
# 3. 然后在对应的环境中直接pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

### 虚拟环境

劝你 还是新建虚拟环境,省去很多麻烦😖

```bash
# 创建新的conda环境（自动确认所有提示）
conda create -n mamba_env python=3.8 -y

# 激活环境
conda activate mamba_env

# 安装兼容版本的PyTorch（自动确认所有提示）
conda install pytorch=2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装Mamba-SSM（自动确认所有提示）
pip install 库名 -y

pip install mamba-ssm --no-cache-dir --verbose
```

### docker

两种解决思路:

(1)mamba 镜像

(2)torch 镜像(因为,没有提供 mamba 镜像,但是报cuda冲突问题)

> 思路是:
>
> - docker pull  拉镜像
> - docker run 挂载 目录
> - 

我放弃了,装不上,使用 docker拉取镜像,挂载目录,问题是服务器上拉不下来,方法是本地 pull,再上传

```bash
# 拉取官方Docker镜像
docker pull statespaces/mamba:latest

# 运行容器并挂载您的代码目录
docker run --gpus all -it -v /home/student2023/xiehr2023/UnetTSF:/workspace statespaces/mamba

# 在容器内运行代码
python /workspace/customLayers/module_4.py
```

关于命令:

```bash
docker run --gpus all -it -v /home/student2023/xiehr2023/UnetTSF:/workspace statespaces/mamba
```

解释

```bash
--gpus all: 允许容器访问所有GPU
-it: 交互式终端
-v /home/student2023/xiehr2023/UnetTSF:/workspace: 将您的本地代码目录挂载到容器内的/workspace目录
statespaces/mamba: 使用官方Mamba镜像
```

常用命令:

```bash
# 查看运行中的容器
docker ps

# 重新连接到运行中的容器（如果您退出了）
docker exec -it 容器ID /bin/bash

# 停止容器
docker stop 容器ID

# 删除容器
docker rm 容器ID
```

持久化:容器保存为镜像,使用镜像挂载目录

```bash
# 将当前容器保存为新镜像
docker commit 容器ID my-mamba-env

# 使用新镜像运行容器
docker run --gpus all -it -v /home/student2023/xiehr2023/UnetTSF:/workspace my-mamba-env
```

我的 mamba 安装(成功版):

```python
conda create -n mamba39 python=3.9 -y
conda activate mamba39
conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda==11.8 -c pytorch -c nvidia -y
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .
```






