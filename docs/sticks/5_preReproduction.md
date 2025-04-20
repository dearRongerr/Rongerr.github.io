# (逐步系列)复现之前

- git clone
- 下载数据集

- conda activate、readme、python 版本

#### 自动确认 y

```bash
# 创建新的conda环境（自动确认所有提示）
conda create -n mamba python==3.8 -y

# 激活环境
conda activate mamba

# 安装兼容版本的PyTorch（自动确认所有提示）
conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda==11.8 -c pytorch -c nvidia -y

### 稳定安装方法# 
# 1. 通过此命令行查看安装的是哪个wheel文件:
pip install mamba-ssm --no-cache-dir --verbose
# 2. 复制给定的.wheel链接到浏览器,直接下载
# 3. 然后在对应的环境中直接
pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```



#### 安装\激活

```python
conda create -n dave python==3.8
conda activate dave
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy
conda install scikit-image
conda install scikit-learn
conda install tqdm
conda install pycocotools
```

#### 进入\退出

```python
# To activate this environment, use               
#     $ conda activate Autoformer
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

#### requirements 安装

```python
conda create -n SegRNN python=3.8
conda activate SegRNN
pip install -r requirements.txt
```

#### 查看虚拟环境列表:

```python
conda env list
conda actiavte 环境名
conda deactivate
```

#### 本地安装:

(1)wheel

- 下载(这种情况发生在我安装 mamba 的时候,下载时,会显示正在下载哪个 wheel 文件,直接将正在下载的 url链接复制到浏览器,会自动下载)
- 进入到有 wheel 文件的文件夹,记得保证在python虚拟环境中,然后,pip install 即可

```bash
pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

- [ ] (2) git安装

```python
# 激活您的环境
conda activate mamba

# 卸载现有版本
pip uninstall -y mamba-ssm

# 克隆仓库
git clone https://github.com/state-spaces/mamba.git
cd mamba

# 安装
pip install -e .
```

> 执行 pip install -e . 时，pip 会在当前目录查找 setup.py 文件,执行该文件中的安装指令
>

- 调试 sh 文件

这里要安装 python 库：

```python
pip install debugpy
```

修改 sh 文件：

```shell
python -u run_longExp.py
# 需要手动点击 调试，开始运行
python -m debugpy --listen 5998 --wait-for-client run_longExp.py 
# 可以移除 --wait-for-client 参数，让程序不必等待客户端连接就开始运行
python -m debugpy --listen 5998 run_longExp.py \
```

修改配置文件  `"configurations"`  

```json
        {
            "name": "[这里更换为任意名称]",
            "type": "debugpy",
            "justMyCode": true,
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5998
            }
        },
```

- 新建python 文件，调试显示形状  

导入

```python
import custom_repr
```



<details>
<summary>  custom_repr.py </summary>


```python
import torch
import pandas as pd
import custom_repr
# -------------------- 自定义包装类 --------------------

class CustomBool:
    def __init__(self, value):
        self.value = bool(value)

    def __repr__(self):
        return f'{{bool}} {self.value}'

class CustomInt:
    def __init__(self, value):
        self.value = int(value)

    def __repr__(self):
        return f'{{int}} {self.value}'

class CustomStr:
    def __init__(self, value):
        self.value = str(value)

    def __repr__(self):
        return f'{{str}} {self.value}'

# 自定义 list 和 dict 子类
class CustomList(list):
    def __repr__(self):
        return f'{{list: {len(self)}}} {super().__repr__()}'

class CustomDict(dict):
    def __repr__(self):
        return f'{{dict: {len(self)}}} {super().__repr__()}'

# 自定义 Tensor 的 __repr__ (Torch)
original_tensor_repr = torch.Tensor.__repr__
def custom_tensor_repr(self):
    return f'{{Tensor: {tuple(self.shape)}}} {original_tensor_repr(self)}'
torch.Tensor.__repr__ = custom_tensor_repr

# 自定义 DataFrame 的 __repr__ (Pandas)
original_dataframe_repr = pd.DataFrame.__repr__
def custom_dataframe_repr(self):
    return f'{{DataFrame: {self.shape}}} {original_dataframe_repr(self)}'
pd.DataFrame.__repr__ = custom_dataframe_repr

# 自定义 DataLoader 的类
class DataLoader:
    def __init__(self, data_size):
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __repr__(self):
        return f'{{DataLoader: {len(self)}}} DataLoader object'

# -------------------- __main__ 函数 --------------------
def main():
    # 使用自定义类型代替原生类型
    my_list = CustomList([1, 2, 3, 4, 5, 6])
    my_dict = CustomDict({'a': 1, 'b': 2, 'c': 3})
    my_bool = CustomBool(True)
    my_int = CustomInt(42)
    my_str = CustomStr("hello")

    # 测试 Tensor
    my_tensor = torch.randn(100, 512)

    # 测试 DataFrame
    my_dataframe = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    # 测试 DataLoader
    my_dataloader = DataLoader(220)

    # 输出内容
    print(my_list)        # {list: 6} [1, 2, 3, 4, 5, 6]
    print(my_dict)        # {dict: 3} {'a': 1, 'b': 2, 'c': 3}
    print(my_bool)        # {bool} True
    print(my_int)         # {int} 42
    print(my_str)         # {str} 'hello'
    print(my_tensor)      # {Tensor: (100, 512)} tensor([...])
    print(my_dataframe)   # {DataFrame: (3, 3)}    A  B  C
    print(my_dataloader)  # {DataLoader: 220} DataLoader object

# 如果是直接运行文件，则调用 main 函数
if __name__ == "__main__":
    main()
```
</p>
</details>

- 删除 .git文件

```python
rm -rf .git
```

- **可选）** 由于是 fork 来的代码，会保留原有作者的版本控制记录

> 拥有一个干净的仓库，并且不会显示原作者的版本记录，同时可以选择地是否从原仓库获取更新。

<u>（1）彻底清除项目的所有之前的提交历史</u>

**完全重新开始，创建全新的仓库**  

```bash
# 1. 备份当前代码
cp -r UnetTSF UnetTSF_backup

# 2. 删除旧的 Git 目录
rm -rf UnetTSF/.git

# 3. 初始化新的 Git 仓库
cd UnetTSF
git init

# 4. 添加所有文件并提交
git add .
git commit -m "Initial commit: Forked from original repo with modifications"
```

**与我自己的远程仓库关联** 

```python
# 添加你自己的远程仓库
git remote add origin https://github.com/你的用户名/仓库名.git
git remote add origin https://github.com/dearRongerr/TSF_Reproduction.git
# 推送代码
git push -u origin main  
```

**如果还想从原始仓库获取更新，保留原始仓库作为上游** 

```bash
# 添加原始仓库作为上游
git remote add upstream https://github.com/原作者/原仓库.git

# 获取上游更新（当需要时）
git fetch upstream
git merge upstream/master  # 或者 main
```

<u>（2）保留完整历史记录：所有之前的提交历史都会保留</u> 

```bash
# 查看当前远程仓库
git remote -v
# 显示结果通常为：
# origin   https://github.com/原作者/原仓库.git (fetch)
# origin   https://github.com/原作者/原仓库.git (push)

# 移除原始的远程仓库引用
git remote rm origin

# 添加你自己的仓库作为新的 origin
git remote add origin https://github.com/你的用户名/你的仓库.git

# 添加原作者的仓库作为 upstream (方便日后同步更新)
git remote add upstream https://github.com/原作者/原仓库.git
```

**更新 .gitignore 文件** 

```bash
# Python 缓存文件
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/

# 虚拟环境
venv/
env/
ENV/

# 分布式训练
*.out
*.err
*.log

# 模型检查点和结果
checkpoints/
results/
*.pth
*.pt
*.bin
*.h5

# 临时文件
.ipynb_checkpoints
.DS_Store
.idea/
.vscode/

# 数据文件 (根据需要调整)
*.csv
*.tsv
*.pkl
*.npy
```

## 模型修改与版本控制



```python
# 进行多次提交开发一个特性
git commit -m "初始化多尺度卷积结构"
git commit -m "添加自注意力机制"
git commit -m "优化特征融合方法"
git commit -m "完成多尺度时间卷积与注意力机制"

# 特性完成后，打一个标签
git tag -a feature-multiscale-complete -m "多尺度时间卷积与注意力机制完整实现"

# 运行实验后，标记结果
git tag -a exp-etth1-96h-mae0.412 -m "ETTh1数据集96小时预测最佳结果"
```

提交&打标签

> 标签是对提交的引用，所以必须先有提交才能打标签

```bash
# 首先提交代码
git add models/Time_Unet.py
git commit -m "完成多尺度时间卷积与注意力机制实现"

# 然后在这个提交上打标签
git tag -a v0.1-multiscale -m "多尺度时间卷积特性完成"
```

