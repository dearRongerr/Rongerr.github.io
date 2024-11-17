# 手撕自注意机制

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

