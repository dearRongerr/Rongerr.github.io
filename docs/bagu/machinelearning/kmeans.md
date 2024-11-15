# 手撕kmeans

```python
import numpy as np
def kmeans(data, k, thresh=1, max_iterations=100):
  centers = data[
     np.random.choice(data.shape[0], k, replace=False)
     ]
  for _ in range(max_iterations):
    distances = np.linalg.norm(
       data[:, None] - centers, 
       axis=2
       ) # n,k,d
    labels = np.argmin(distances, axis=1)
    new_centers = np.array(
       [data[labels == i].mean(axis=0) for i in range(k)]
       )
    if np.all(centers == new_centers):break
    center_change = np.linalg.norm(new_centers - centers)
    if center_change < thresh:break
    centers = new_centers
  return labels, centers
data = np.random.rand(100, 2)  # 100个样本，每个样本有两个特征
k = 3  # 聚类数为3
labels, centers = kmeans(data, k)
print("簇标签:", labels)
print("聚类中心点:", centers)
```

