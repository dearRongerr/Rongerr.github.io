# 手撕Transformer代码

## 自注意力机制

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self,model_dim,dropout=0.1):

        super().__init__()
        self.model_dim = model_dim

        self.q_proj = nn.Linear(model_dim,model_dim)
        self.k_proj = nn.Linear(model_dim,model_dim)
        self.v_proj = nn.Linear(model_dim,model_dim)

        self.o_proj = nn.Linear(model_dim,model_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q,k.transpose(-1,-2))//math.sqrt(self.model_dim)

        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e09)

        prob = F.softmax(scores,dim=-1)

        prob = self.dropout(prob)

        attn_weights = torch.matmul(prob,v)

        output = self.o_proj(attn_weights)

        return output

model_dim = 512
seq_len = 8
batch_size = 2
# mask shape = seq_len * model_dim
x = torch.randn(batch_size,seq_len,model_dim)

sa = SelfAttention(model_dim)

attn_weights = sa(x)
print(attn_weights.shape)  # 输出的形状应该是(batch_size, seq_len, model_dim)
```

## 手撕多头注意力机制

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,model_dim,num_heads,dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim,model_dim)
        self.k_proj = nn.Linear(model_dim,model_dim)
        self.v_proj = nn.Linear(model_dim,model_dim)

        self.dropout = nn.Dropout(dropout)

        self.o_proj = nn.Linear(model_dim,model_dim)
    
    def forward(self,q,k,v,mask=None):

        batch_size,sequence_length,model_dim = q.shape

        q = self.q_proj(q).view(batch_size,sequence_length,self.num_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(k).view(batch_size,sequence_length,self.num_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(v).view(batch_size,sequence_length,self.num_heads,self.head_dim).transpose(1,2)

        scores = torch.matmul(q,k.transpose(-1,-2))//math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e09)

        prob = F.softmax(scores,dim=-1)

        prob = self.dropout(prob)

        attn_weights = torch.matmul(prob,v).transpose(1,2).contiguous().view(batch_size,sequence_length,model_dim)

        output = self.o_proj(attn_weights)
        return output

model_dim = 512 
num_heads = 8
mha = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=0.1)
batch_size = 10
sequence_length = 60
q = torch.randn(batch_size, sequence_length, model_dim)
k = torch.randn(batch_size, sequence_length, model_dim)
v = torch.randn(batch_size, sequence_length, model_dim)

mask = None
output = mha(q, k, v, mask)
print(output.shape)  # 输出的形状应该是(batch_size, sequence_length, model_dim)
```

note：

- 假设输入数据q, k, v的形状是(batch_size, sequence_length, model_dim)

- 例如，一个批次大小为10，序列长度为60，模型维度为512的输入 

- 这是因为注意力机制的目的是为序列中的每个元素生成一个新的表示，
  而不是生成一个序列长度为 sequence_length 的序列。

​	注意力机制为序列中的每个元素生成一个新的表示，这个表示综合了序列中所有元素的信息，

​	但输出的形状仍然是 (batch_size, sequence_length, model_dim)，

​	而不是 (batch_size, sequence_length, sequence_length)。

​	这是因为输出的每个元素是序列中对应位置的元素在所有头中的加权表示，
​	而不是序列中每个元素对其他元素的注意力权重矩阵。

## 手撕位置编码



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### 一维绝对位置编码

- 类写法
- 函数写法

```python
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

函数写法

```python
def create_1d_absolute_sincos_embeddings(max_sequence_length,model_dim):
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
    pe_table = create_1d_absolute_sincos_embeddings(max_sequence_length,model_dim)
    print(pe_table)
```

### 一维可学习绝对位置编码

from vit

类写法：

```python
import torch
import torch.nn as nn
class TrainablePositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model):
        super().__init__() 
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        pe = nn.Embedding(self.max_sequence_length, self.d_model)
        nn.init.constant_(pe.weight, 0.)
        return pe
max_sequence_length = 100
d_model = 512
trainable_pe = TrainablePositionEncoding(max_sequence_length, d_model)
position_encodings = trainable_pe()
print(position_encodings.weight.shape)  # 输出：torch.Size([100, 512])
```

函数写法：

```python
def create_1d_absolute_trainable_embeddings(max_sequence_length,model_dim):
    pe = nn.Embedding(max_sequence_length,model_dim)
    nn.init.constant_(pe.weight,0.)

    return pe
```

### 二维相对可学习位置编码

from SwinTransformer

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

### 二维绝对位置编码

from MAE

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

