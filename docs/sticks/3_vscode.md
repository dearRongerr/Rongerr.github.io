# VSCode

## vscode 注释关键字高亮

- 【参看】

[vscode设置关键字高亮](https://www.cnblogs.com/penuel/p/17442855.html)

`command+shift+p`打开命令窗口，搜索`Highlight Toggle Current`命令，此命令可高亮选中的单词

```
Highlight Toggle Current
```

即可设置：

![image-20250303212556242](images/image-20250303212556242.png)

![image-20250303213208276](images/image-20250303213208276.png)

效果：

![image-20250303213220285](images/image-20250303213220285.png)

- command + ， 搜索 highlight代码编辑器位置，一样打开

## vscode 调试 shell 调用的 python 文件

[vscode 中如何Debug用Bash脚本运行的Python代码](https://blog.csdn.net/m0_52394190/article/details/136913701)

[使用VSCode对一个用Shell脚本启动的Python程序进行Debug](https://www.cnblogs.com/mactor/p/18476063)

[使用VSCode调试带参数的Python脚本](https://www.bilibili.com/opus/686045842139774999)

（1）激活python 运行相应的虚拟环境

（2）安装 `debugpy` 库

```
pip install debugpy
```

（3）查看可用端口：

```
for port in {5000..6000}; do
    (echo > /dev/tcp/localhost/$port) >/dev/null 2>&1 || echo "$port is free"
done
```

查看5000-6000之间的可用端口，用的 `5999`

（4）launch.json 配置

在VSCode按`Command+Shift+P`, 输入`Debug: Add Configuration` 新建一个launch.json（或者直接打开launch.json)编辑以下内容并保存

```
{
	"version": "0.2.0",
	"configurations": [
		{
			"name": "[这里更换为任意名称]",
			"type": "python",
			"request": "attach",
			"connect": {
				"host": "localhost",
				"port": 5999
			}
		}
	]
}
```

（5）Shell
使用Shell 启动Python，对原来的Shell进行编辑，
例如原来的Shell是：

```
python [NAME].py --arg1 "arg" --arg2 "123" 
```

只需要在`python`和`[NAME].py`之间加入 `-m debugpy --listen [端口号]`

即：

```
python -m debugpy --listen 5999 [NAME].py --arg1 "arg" --arg2 "123" 
```

（6）运行

在运行的 python 的文件中，打上相应的断点，然后运行 shell 文件

```
sh your_shell_path/shell_name.sh
```

即可激活 vscode 的相应调试

![image-20250301180429828](images/image-20250301180429828.png)

==一些不太懂的设置：==

- 应用市场扩展：Bash Debug

- `launch.json`的配置：点击 `添加配置`，新增 3 个：

（1）Bash-Debug (type in script name)

（2）Bash-Debug (hardcoded script name)

（3）Bash-Debug (simplest configuration)



```
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
			"name": "[这里更换为任意名称]",
			"type": "python",
			"request": "attach",
			"connect": {
				"host": "localhost",
				"port": 5999
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



## 调试不显示张量形状问题

【参看】[在 VS Code 中调试 Tensor 形状不显示的问题及解决方案](https://blog.csdn.net/weixin_51524504/article/details/143101401?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ECtr-5-143101401-blog-139059249.235%5Ev43%5Epc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ECtr-5-143101401-blog-139059249.235%5Ev43%5Epc_blog_bottom_relevance_base4&utm_relevant_index=10)

![image-20250304135719100](images/image-20250304135719100.png)



```
import torch
import pandas as pd

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

