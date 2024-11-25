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

注释：

```python
import numpy as np
def kmeans(data, k, thresh=1, max_iterations=100):
  # data ： n_samples * feature_dim
  # 从 data 中随机选择 n_clusters 个不同的样本索引
  # centers : n_clusters * feature_dim
  # replace=False：表示在选择样本时不允许重复，即每个样本只能被选择一次。
  centers = data[np.random.choice(data.shape[0], k, replace=False)]
  for _ in range(max_iterations):
    # data ： n_smaples * feature_dim
    # data[:, None]：n_samples * 1 * feature_dim  \ n 1 d
    # centers ： n_clusters * feature_dim \ k d
    # data[:, None] - centers : n_samples * n_clusters * feature_dim \ n k d
    # np.linalg.norm : n_samples * n_clusters
    distances = np.linalg.norm(data[:, None] - centers, axis=2)
    # labels : n_samples * 1
    # distances : n_samples * n_clusters
    # np.argmin 函数返回指定轴上最小值的索引。
    # axis=1 表示沿着第 1 轴（即列）寻找最小值的索引。
    labels = np.argmin(distances, axis=1)
    '''
        距离矩阵:
        [[0.5 1.2 0.9]
        [1.  0.8 1.5]
        [0.3 0.4 0.2]
        [1.1 0.7 0.6]
        [0.9 1.3 0.4]]
        簇标签:
        [0 1 2 2 2]
        簇标签的形状: (5,)
    '''
    new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    
    # np.all(centers == new_centers) 返回 True，输出结果为 "中心点已收敛"。
    if np.all(centers == new_centers):
      break
    # 计算 new_centers 和 centers 之间的欧几里得距离（或范数）。
    center_change = np.linalg.norm(new_centers - centers)
    if center_change < thresh:
        break
    centers = new_centers
  return labels, centers

data = np.random.rand(100, 2)  # 100个样本，每个样本有两个特征
k = 3  # 聚类数为3
labels, centers = kmeans(data, k)
print("簇标签:", labels)
print("聚类中心点:", centers)
```

