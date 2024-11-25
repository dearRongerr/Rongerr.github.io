# 4种位置编码

- [x] 一维绝对位置编码
- [x] 一维可学习位置编码
- [x] 二维相对偏置位置编码
- [x] 二维绝对位置编码
- [ ] 旋转位置编码

## 位置编码为什么是三角函数形式的？

1. 最直观的编码方式是从0到sequence length，但是无界
2. 用 $\frac{1}{sequence\_length}$ 改变了词与词之间的相对位置
3. 二进制编码，d model通常设置为512,2的512次方能编码完 max sequence length个位置，但是是离散的
4. 连续，带有周期性的三角函数位置编码，类似二进制，低位变化快，高位变化慢



----

[【46、四种Position Embedding的原理与PyTorch手写逐行实现（Transformer/ViT/Swin-T/MAE）-哔哩哔哩】]( https://b23.tv/is1p6aN)

![image-20241117170228318](images/image-20241117170228318.png)

## 原始Transformer的位置编码 ：一维绝对、常数位置编码

![image-20241117170648419](images/image-20241117170648419.png)

pos：句子中词的位置（0-max sequence length）

i：词嵌入位置（0—255）

1D绝对位置编码，常数不需要训练

代码实现：

(类写法)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinCosPositionEmbedding(nn.Module):
    def __init__(self, max_sequence_length,model_dim):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.model_dim = model_dim
    def forward(self):
        pe = torch.zeros(self.max_sequence_length,self.model_dim)
        pos_mat = torch.arange(self.max_sequence_length).reshape(-1,1)
        i_mat = torch.pow(10000,
                          torch.arange(0,self.model_dim,2).reshape(1,-1)/self.model_dim
                          )
        
        pe[:,0::2] = torch.sin(pos_mat/i_mat)
        pe[:,1::2] = torch.cos(pos_mat/i_mat)

        return pe
print(SinCosPositionEmbedding(max_sequence_length=8,model_dim=4).forward())
```

（函数写法）

```python
def position_sincos_embedding(max_sequence_length,model_dim):
    assert model_dim%2 == 0,"wrong dimension"
    pe_table = torch.zeros(max_sequence_length,model_dim)
    pos_mat = torch.arange(max_sequence_length).reshape(-1,1)
    i_mat = torch.pow(
        10000,
        torch.arange(0,model_dim,2)/model_dim
    )
    pe_table[:,0::2]=torch.sin(pos_mat/i_mat)
    pe_table[:,1::2]=torch.cos(pos_mat/i_mat)
    return pe_table

# Transformer论文 一维绝对位置编码
if __name__=="__main__":
    max_sequence_length = 8
    model_dim = 4
    pe_table = position_sincos_embedding(max_sequence_length,model_dim)
    print(pe_table)
```

## ViT  1维绝对的可学习的位置编码

![image-20241117173149421](images/image-20241117173149421.png)

标准的、可学习的一维位置编码；二维的位置编码并没有带来更好的效果

```python
def create_1d_absolute_trainable_embeddings(max_sequence_length,model_dim):
    pe = nn.Embedding(max_sequence_length,model_dim)
    nn.init.constant_(pe.weight,0.)

    return pe
```



## SwinTransformer 2维的、相对的、基于位置偏差可训练的位置编码

![image-20241117174359681](images/image-20241117174359681.png)

- 相对位置编码、可学习的、相对位置偏差加到每一个头上
- $QK^T$ 的维度是 $序列长度 × 序列长度$，所以B的形状也是  $序列长度 × 序列长度$
- 考虑head，那么形状是 $num\_head \times L \times L$
- 由于是可学习的，要计算两两Patch的偏差，$Position\_bias$，把$bias$当成索引，从$bias\_matrix$里查找到$learnable \_ vector$，即可学习的向量
- 可以看到 偏差矩阵是 $\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}$

代码：

- 首先，由于是二维的，所以既要考虑横轴，又要考虑纵轴 

- 二维、相对的、基于bias的、可训练的位置编码 `create_2d_relative_bias_trainable_embeddings`

```python
def create_2d_relative_bias_trainable_embeddings(n_head,height,width,dim):
    # width:5,[0,1,2,3,4],bias=[-width+1,width-1],2*width-1
    # height:5,[0,1,2,3,4],bias=[-height+1,height-1],2*height-1

    position_embedding = nn.Embedding((2*width-1)*(2*height-1),n_head)
    nn.init.constant_(position_embedding.weight,0.)

    def get_relative_position_index(height,width):
        m1,m2 = torch.meshgrid(torch.arange(height),torch.arange(width))
        coords = torch.stack(m1,m2) #[2,height,width]
        coords_flatten = torch.flatten(coords,1) #[2,height*width]

        # 把偏差变成正数，然后从position_embedding中按索引取值
        relative_coords_bias = coords_flatten[:,:,None]-coords_flatten[:,None,:] # [2,height*width,height*width]

        relative_coords_bias[0,:,:] += height-1
        relative_coords_bias[1,:,:] += width-1

        # A:2d,B:1d,B[[i*cols+j] = A[i,j]
        relative_coords_bias[0,:,:] *= relative_coords_bias[1,:,:].max()+1

        return relative_coords_bias.sum(0) # [height*width,height*width]
    relative_position_bias = get_relative_position_index(height,width)
    bias_embedding = position_embedding(torch.flatten(relative_position_bias)).reshape(height*width,height*width,n_head) #[height*width,height*width,n_head]

    bias_embedding = position_embedding.permute(2,0,1).unsqueeze(0) # [1,n_head,height*width,height*width]

    return bias_embedding
```



## MAE中的位置编码

附录部分

![image-20241125101957652](images/image-20241125101957652.png)

- sin、cos位置编码
- 没有相对位置或者Layer scaling
- 二维的、绝对的 sin cos embedding

```python
# 4.2d absolute constant sincos embedding
# Masked AutoEncoder 论文
def create_2d_absolute_sincos_embeddings(height,width,dim):
    assert dim%4 ==0,"wrong dimension!"
    position_embedding = torch.zeros(height*width,dim)
    m1,m2 = torch.meshgrid(torch.arrange(height,dtype=torch.float),torch.arrange(width,dtype=torch.float))
    coords = torch.stack(m1,m2)  # [2,height*width]
    
    height_embedding = create_1d_absolute_sincos_embeddings(torch.flatten(coords[0]),dim//2)  # [height*width,dim//2]
    width_embedding = create_1d_absolute_sincos_embeddings(torch.flatten(coords[1]),dim//2)  # [height*width,dim//2]

    position_embedding[:,:dim//2] = height_embedding
    position_embedding[:,:dim//2] = width_embedding

    return position_embedding
```

全部代码

![image-20241125103456704](images/image-20241125103456704.png)

![image-20241125103519421](images/image-20241125103519421.png)

![image-20241125103552862](images/image-20241125103552862.png)

![image-20241125103632410](images/image-20241125103632410.png)

- [ ] 旋转位置编码
