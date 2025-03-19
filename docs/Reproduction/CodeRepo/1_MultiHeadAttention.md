# 多头注意力机制形状变化

## QKV

关于 Transformer 的 QKV 同源问题

$QK^TV = B_QL_QD_Q \cdot B_KD_KL_K \cdot B_VL_VD_V$

（1）回答同源问题：

- 自注意力机制中（selfA），$QKV$ 来自同一个序列
- 交叉注意力机制中（crossA），$Q$ 来自一个序列，$KV$ 来自另一个序列，捕捉两个不同序列之间的依赖关系

（2）讨论形状问题：

- 必须满足的是： $D_Q = D_K$ 、 $L_K = L_V$
- 简化版本： $attn = L_Q \times L_K$

---

**多头注意力机制：**

的形状变化：BLD  $\stackrel{D=H\cdot d}{\rightarrow}$  BLHd 

---

## [**形状变化**](#forward) 

```python
def forward(self, q, k, v, mask=None):
    batch_size, sequence_length, model_dim = q.shape

    # [B, L, d_model] → 线性投影 → [B, L, H*d_k] → view → [B, L, H,d_k] → transpose  → [B, H, L, d_k]
    q = self.q_proj(q).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
    # [B, L, d_model] → 线性投影 → [B, L, H*d_k] → view → [B, L, H,d_k] → transpose  → [B, H, L, d_k]
    k = self.k_proj(k).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
    # [B, L, d_model] → 线性投影 → [B, L, H*d_v] → view → [B, L, H,d_v] → transpose  → [B, H, L, d_v]
    v = self.v_proj(v).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

    # [B, H, L, d_k] × [B, H, d_k, S] → 矩阵乘法(torch.matmul) → [B, H, L, S]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e09)

    # [B, H, L, S] → softmax → [B, H, L, S]
    prob = F.softmax(scores, dim=-1)

    prob = self.dropout(prob)

    # [B, H, L, S] × [B, H, S, d_v] → 矩阵乘法(torch.matmul) → [B, H, L, d_v] → 转置(.transpose) → [B, L, H, d_v] → 重塑(.view) → [B, L, d_model]
    attn_weights = torch.matmul(prob, v).transpose(1, 2).contiguous().view(batch_size, sequence_length, model_dim)

    # 恢复原始维度：[B, L, d_model] → 线性投影 → [B, L, d_model]
    output = self.o_proj(attn_weights)
    return output
```

<a name="forward">形状分析 </a> 

（1）**查询（queries）**：

- `[B, L, d_model] → 线性投影(nn.Linear) → [B, L, H*d_k] → 重塑(.view) → [B, L, H, d_k] → 转置(.transpose) → [B, H, L, d_k]`
- 通过线性投影将查询向量投影到多头注意力空间，并重塑为多头的形状。

```python
q = self.q_proj(q).view(batch_size,sequence_length,self.num_heads,self.head_dim).transpose(1,2)
```



（2）**键（keys）**：  $D_Q = D_K \triangleq d_k$    

- `[B, S, d_model] → 线性投影(nn.Linear) → [B, L, H*d_k] → 重塑(.view) → [B, L, H, d_k] → 转置(.transpose) → [B, H, L, d_k]`
- 通过线性投影将键向量投影到多头注意力空间，并重塑为多头的形状。
- 更准确的记号，表示 QK的维度必须相同 $\triangleq d_k$ ，KV 的长度必须相同 $\triangleq S$

> 简化版
>
> -  $Q \triangleq BLD_k、  K \triangleq BSD_k 、V \triangleq BSD_v$ 
> - ==linear 、view、transpose==  $Q：BLD_k -> BLHd_k -> BHLd_k$ 
> - $K：BSD_k -> BSHd_k -> BHSd_k$ 
> - $V：BSD_v -> BSHd_v -> BHSd_v$
> -  ==torch.matmal==  $QK^T: BHLd_k \cdot BHd_kS -> BHLS$
> -  ==softmax 、dropout、torch.matmul== $sonftmax(QK^T)V ->BHLS \cdot BHSd_v -> BHLd_v -> transpose -> view -> BLD$  

（3）**值（values）**：

- `[B, L, d_model] → 线性投影(nn.Linear) → [B, L, H*d_v] → 重塑(.view) → [B, L, H, d_v] → 转置(.transpose) → [B, H, L, d_v]`
- 通过线性投影将值向量投影到多头注意力空间，并重塑为多头的形状。

（4）**计算注意力得分**：

```python
scores = torch.matmul(q,k.transpose(-1,-2))//math.sqrt(self.head_dim)
```

- `[B, H, L, d_k] × ([B, H, L, d_k] → transpose → [B, H, d_k, L]) → 矩阵乘法(torch.matmul) → [B, H, L, L]`  
- 通过矩阵乘法计算查询和键之间的点积注意力得分，并进行缩放。

（5）**计算注意力加权值**：

- `[B, H, L, L] → softmax → [B, H, L, L]`
- 通过softmax计算注意力权重。

（6）**应用注意力权重**：

- `[B, H, L, L] × [B, H, L, d_v] → 矩阵乘法 → [B, H, L, d_v] → 转置 → [B, L, H, d_v] → 重塑 → [B, L, d_model]`
- 通过矩阵乘法将注意力权重应用于值向量，并重塑为原始形状。

（7）**输出**： 还原维度



- `[B, L, d_model] → 线性投影 → [B, L, d_model]`
- 通过线性投影将合并的输出恢复到原始维度。

📢 简化版

-  $Q \triangleq BLD_k、  K \triangleq BSD_k 、V \triangleq BSD_v$ 
- ==linear 、view、transpose==  $Q：BLD_k -> BLHd_k -> BHLd_k$ 
- $K：BSD_k -> BSHd_k -> BHSd_k$ 
- $V：BSD_v -> BSHd_v -> BHSd_v$
-  ==torch.matmal==  $QK^T: BHLd_k \cdot BHd_kS -> BHLS$
-  ==softmax 、dropout、torch.matmul== $sonftmax(QK^T)V ->BHLS \cdot BHSd_v -> BHLd_v -> transpose -> view -> BLD$  

完整的代码，得会呀

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

