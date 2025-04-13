# 复现之前

- git clone
- 下载数据集

- conda activate、readme、python 版本

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

```python
# To activate this environment, use               
#     $ conda activate Autoformer
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

```python
conda create -n SegRNN python=3.8
conda activate SegRNN
pip install -r requirements.txt
```

```python
conda env list
conda actiavte 环境名
conda deactivate
```

- 调试 sh 文件

修改 sh 文件：

```shell
python -u run_longExp.py
python -m debugpy --listen 5998 --wait-for-client run_longExp.py 
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
<summary>python代码</summary>


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