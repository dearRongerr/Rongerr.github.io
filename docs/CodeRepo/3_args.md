# 类的初始化传入命令行参数

主函数传入命令行参数，调用的类就能调用这个命令行参数

```python
import argparse

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=30)
    img_root = 'http://... or 本地路径'
    parser.add_argument("--data_path",type=str,default=img_toor)
    parser.add_argument("device",default="cuda",help="判断当前设备是否能使用 cuda")
    
    opt = parser.parse_args()
    main(opt) # 传的参数是实例化的类
```

> 场景描述：如果只是想简单的测试 使用命令行参数的类呢？避免重复的从大项目的 main 开始执行

```python
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.input_channels = configs.enc_in
        self.input_len = configs.seq_len
        self.out_len = configs.pred_len
        self.individual = configs.individual
        # 下采样设定
        self.stage_num = configs.stage_num
    def forward(self, x):
        '''
            [B,T,C] -> [B,P,C]
        
        '''
        x = self.revin_layer(x, 'norm')  # [B,T,C]
        return x
    
if __name__ == '__main__':    
    # 创建一个简单的配置对象
    class SimpleConfig:
        def __init__(self):
            self.enc_in = 7        # ETT数据集特征维度
            self.seq_len = 96      # 输入序列长度
            self.pred_len = 720    # 预测序列长度
            self.individual = True # 是否独立处理每个通道
            self.stage_num = 4     # U-Net的深度
            self.stage_pool_kernel = 3  # 池化核大小
            self.stage_pool_stride = 2  # 池化步长
            self.stage_pool_padding = 1 # 池化填充
            self.trend_kernel = 13  # 趋势分解窗口大小
    
    # 创建配置
    configs = SimpleConfig()
    
    # 无需传递的参数，作用于仅在当前函数
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 通过实例化的类 索引参数，相当于定义当前作用于的变量值
    seq_len = configs.seq_len
    enc_in = configs.enc_in
    
    # 实例化模型
    model = Model(configs).to(device)
    # 生成随机输入数据
    x = torch.randn(batch_size, seq_len, enc_in).to(device)
    # 前向传播
    output = model(x)    
    print("模型结构:",model)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"预期输出形状: [batch_size={batch_size}, pred_len={configs.pred_len}, enc_in={enc_in}]")
```

- 想说的是，可以简单的定义一个配置类 `SimpleConfig()` ，将这个类的实例 `configs=SimpleConfig()` ，传入初始化需要命令行参数的类  `model = Model(configs)`
- 这个简单的配置类不接收任何参数，只需要一个 init 定义参数即可，实现在类与类之间传递参数，避免初始化的时候，需要一大堆参数，需要哪个 `configs.`  索引即可

或者就这样：（其实我也还是有点晕

```python
import torch
import torch.nn as nn
import argparse
import sys

# 简化的模型定义
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_channels = configs.enc_in
        self.input_len = configs.seq_len
        self.out_len = configs.pred_len
        self.individual = configs.individual
        self.stage_num = configs.stage_num
        # 简化起见，只定义一个简单的层
        self.revin_layer = nn.LayerNorm([self.input_len, self.input_channels])
    
    def forward(self, x):
        '''
            [B,T,C] -> [B,P,C]
        '''
        x = self.revin_layer(x)  # [B,T,C]
        # 简化的前向传播，只返回一个调整形状的张量模拟输出
        return x[:, :self.out_len, :]



if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Time-Unet模型测试')
    
    # 添加命令行参数
    parser.add_argument('--enc_in', type=int, default=7, help='输入特征维度')
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=720, help='预测序列长度')
    parser.add_argument('--individual', action='store_true', help='是否独立处理每个通道')
    parser.add_argument('--stage_num', type=int, default=4, help='U-Net深度')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--print_model', action='store_true', help='是否打印模型结构')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 实例化模型并移至指定设备
    model = Model(args).to(device)
    
    # 生成随机输入数据
    x = torch.randn(args.batch_size, args.seq_len, args.enc_in).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 打印信息
    if args.print_model:
        print("模型结构:", model)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"预期输出形状: [batch_size={args.batch_size}, pred_len={args.pred_len}, enc_in={args.enc_in}]")
```

该说不说，封装的好抽象：

```python
import argparse
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    简单的神经网络模型，继承自nn.Module
    用于演示模块化代码结构
    """
    def __init__(self, args):
        """
        模型初始化
        参数:
            args: 包含模型配置的参数对象
        """
        super(Model, self).__init__()
        # 从args获取配置参数
        self.input_channels = args.enc_in
        self.input_len = args.seq_len
        self.out_len = args.pred_len
        self.individual = getattr(args, 'individual', False)
        
        # 定义网络层
        self.norm_layer = nn.LayerNorm([self.input_len, self.input_channels])
        self.fc = nn.Linear(self.input_len, self.out_len)
        
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量, 形状为 [batch_size, seq_len, enc_in]
        返回:
            输出张量, 形状为 [batch_size, pred_len, enc_in]
        """
        # 简单处理: 归一化、转置、线性变换、再转置
        x = self.norm_layer(x)  # [batch, seq_len, enc_in]
        x_t = x.transpose(1, 2)  # [batch, enc_in, seq_len]
        out = self.fc(x_t)  # [batch, enc_in, pred_len]
        return out.transpose(1, 2)  # [batch, pred_len, enc_in]

def parse_args():
    """
    解析命令行参数
    
    返回:
        args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='模块化模型测试示例')
    
    # 添加命令行参数
    parser.add_argument('--enc_in', type=int, default=7, help='输入特征维度')
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=48, help='预测序列长度')
    parser.add_argument('--individual', action='store_true', help='是否独立处理每个通道')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='计算设备')
    parser.add_argument('--print_model', action='store_true', help='是否打印模型结构')
    
    return parser.parse_args()

def build_model(args):
    """
    构建并初始化模型
    
    参数:
        args: 包含模型配置的参数对象
    
    返回:
        model: 构建的模型
        device: 计算设备
    """
    device = torch.device(args.device)
    model = Model(args).to(device)
    print(f"模型已构建，使用设备: {device}")
    return model, device

def generate_data(args, device):
    """
    生成测试数据
    
    参数:
        args: 包含数据配置的参数对象
        device: 计算设备
    
    返回:
        data: 生成的随机数据
    """
    data = torch.randn(args.batch_size, args.seq_len, args.enc_in).to(device)
    print(f"已生成随机数据，形状: {data.shape}")
    return data

def evaluate_model(model, data):
    """
    模型评估
    
    参数:
        model: 待评估的模型
        data: 输入数据
    
    返回:
        output: 模型预测结果
    """
    print("开始模型评估...")
    with torch.no_grad():
        output = model(data)
    print("评估完成！")
    return output

def print_results(args, data, output):
    """
    打印结果
    
    参数:
        args: 参数对象
        data: 输入数据
        output: 模型输出
    """
    print("\n----- 结果报告 -----")
    print(f"输入形状: {data.shape}")
    print(f"输出形状: {output.shape}")
    print(f"预期输出形状: [batch_size={args.batch_size}, pred_len={args.pred_len}, enc_in={args.enc_in}]")
    print("-------------------\n")

def main():
    """
    主函数：协调整个程序的执行流程
    """
    print("===== 开始执行模型测试 =====\n")
    
    # 步骤1: 解析参数
    print("步骤1: 解析命令行参数")
    args = parse_args()
    
    # 步骤2: 构建模型
    print("\n步骤2: 构建模型")
    model, device = build_model(args)
    
    # 步骤3: 生成数据
    print("\n步骤3: 生成测试数据")
    data = generate_data(args, device)
    
    # 步骤4: 评估模型
    print("\n步骤4: 评估模型")
    output = evaluate_model(model, data)
    
    # 步骤5: 打印结果
    print("\n步骤5: 打印结果")
    if args.print_model:
        print("模型结构:", model)
    print_results(args, data, output)
    
    print("===== 测试完成 =====")

if __name__ == '__main__':
    main()
```

也还行吧。