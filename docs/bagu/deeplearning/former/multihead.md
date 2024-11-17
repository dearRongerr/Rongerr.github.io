# 手撕多头自注意力机制

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