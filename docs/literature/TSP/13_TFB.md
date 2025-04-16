# TFB

TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

（CCF A类会议 VLDB2024 最佳研究论文奖提名）

作者：23 级华东师范大学研究生

> International Conference on Very Large Data Bases（数据库三大顶会之一：VLDB、SIGMOD、ICDE）

作者给出的中文版解读：[https://zhuanlan.zhihu.com/p/695413738](https://zhuanlan.zhihu.com/p/695413738)

原文：[https://www.vldb.org/pvldb/vol17/p2363-hu.pdf](https://www.vldb.org/pvldb/vol17/p2363-hu.pdf) 

源码：[https://github.com/decisionintelligence/TFB](https://github.com/decisionintelligence/TFB)

主页：[https://decisionintelligence.github.io/OpenTS/](https://decisionintelligence.github.io/OpenTS/)

视频讲解：[论文研读之时序预测基准TFB](https://www.bilibili.com/video/BV1fYH4eQEPv/?spm_id_from=333.337.search-card.all.click&vd_source=ddd7d236ab3e9b123c4086c415f4939e)

同团队文章 **FOUNDTS**：[https://arxiv.org/pdf/2410.11802](https://arxiv.org/pdf/2410.11802)

----

本文关键词：

- 目的：benchmark
-  8,068 time series ，25 multivariate time series
- Statistical Learning (SL)、Machine Learning(ML)、Deep Learning (DL)、

> 涉及不同的年周期、月周期，具有不同的趋势
>
> 测评了传统时序方法、机器学习方法、现代深度学习方法（会引出一个思考，复杂的就是好吗

[实验结论](https://zhuanlan.zhihu.com/p/708318218)：

作者对TFB中包含的所有数据集，包括25个多变量数据集和8,068个单变量时间序列，以及前文提到的所有baseline方法，进行细致的实验分析，限于篇幅不在展示。我比较关心的一些结论：

- **线性模型**在数据集==呈增长趋势或具有显著漂移==时表现出色。这可以归因于线性模型的线性建模能力，使其能够很好地捕捉线性趋势和漂移。
- **Transformer方法** 在展现明显==季节性、平稳性和非线性模式==，以及更明显模式或内在相似性的数据集上优于线性方法。这种优越性可能源于Transformer方法增强的非线性建模能力。

==图 1==

![image-20250415161733250](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504151617792.png)

时间序列数据不同特性可视化，说明什么是季节性、趋势性、偏移性、平稳性和转移

图1包含12个子图，每个子图代表一种数据特征的可视化：

(a) Seasonality（季节性）：显示了具有明显周期性波动的数据。

(b) Trend（趋势）：展示了数据随时间呈现的长期增长或下降趋势。

(c) Shifting（变化）：数据在某个时间点出现了明显的水平变化。

(d) Stationarity（平稳性）：数据的统计特性（如均值和方差）随时间保持不变。

(e) Transition（转换）：数据在某个时间点出现了突变，可能是由于外部事件或系统内部变化引起的。

(f) Non-Seasonality（非季节性）：数据没有明显的周期性波动。

(g) Non-Trend（非趋势）：数据没有明显的长期增长或下降趋势。

(h) Non-Shifting（非变化）：数据没有出现明显的水平变化。

(i) Non-Stationarity（非平稳性）：数据的统计特性随时间变化。

(j) Non-Transition（非转换）：数据没有出现明显的突变。

每个子图的右上角都有一个小的数值框，显示了该特征的某种统计度量（例如，季节性强度、趋势斜率、变化幅度、平稳性测试的p值等）

