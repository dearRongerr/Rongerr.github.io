# TSP

> [ML 领域三大顶刊](https://www.bilibili.com/video/BV1Nr421p7NV/?spm_id_from=333.788.top_right_bar_window_history.content.click&vd_source=ddd7d236ab3e9b123c4086c415f4939e) 
>
> - NIPS（暑假要开始，收稿量非常大，同时质量非常过硬）——OpenReview
> - ICLR（暑假刚结束）——OpenReview
> - ICML（寒假结束）
>
> 关于分类：
>
> - Oral：12 分钟口头陈述
> - Splotlights（特别关注）：4 分钟的口头演示
> - Posters（海报）：其余被接收的论文都是海报演示
> - ORALS >  Splotlights ＞ POSTERS

- [ ] NeurIPS2019、LogTrans
- [ ] ICLR2020、Reformer

---

- [x] NeurIPS2021 、Autoformer、清华大学吴海旭
- [x] AAAI2021(Best Paper)、Informer

---

- [x] ICLR2022  (Oral)、Pyraformer、上海交通大学、蚂蚁集团
- [x] ICML2022、Fedformer、阿里达摩院
- [ ] IJCAI2022、Triformer
- [ ] NeuraIPS2022、SCINet

---



- [x] AAAI2023 、 LTSF-Linear(DLinear、NLinear)
- [ ] NeurIPS2023 、TLNets
- [x] NeurIPS2023 (Spotlight)、WITRAN、北京交通大学万怀宇团队
- [ ] ICLR2023、Crossformer
- [ ] 2023、TimesNet
- [ ] ICLR2023、PatchTST
- [x] ICLR2023、SegRNN、华南理工大学

----

- [x] ICLR2024、TimeMixer、蚂蚁集团、清华大学吴海旭
- [ ] ICLR 2024、、iTransformer、
- [x] 2024、UnetTSF、中国科学技术大学、中国科学院
- [x]  VLDB2024（CCF-A）、TFB、华东师范大学决策智能实验室，华为云

----

PaperWithCode：[Time Series Forecasting ](https://paperswithcode.com/task/time-series-forecasting) 

- 特征之间的相关性、通道独立
- 注意力机制、复杂度

## 论文关键词

> former 系、linear 系、RNN 系分别适用于什么情况？（TFB 中做了详细的实验，有结论。
>
> - [former 系适用于周期性明显的数据](https://www.bilibili.com/video/BV1fYH4eQEPv/?spm_id_from=333.337.search-card.all.click&vd_source=ddd7d236ab3e9b123c4086c415f4939e)
> - 

### Informer

概率稀疏自注意力

### Autoformer

引入序列分解&自相关机制

基本遵循原始 Transformer 的设计 框架

应用于2022北京冬奥会 10 分钟天气预测、清华软院

### Pyraformer

金字塔注意力，很复杂，有证明，引入了新概念：最大信号传递路径

### Fedformer

频域信息，傅里叶&小波，改进Autoformer、专家混合分解，有频域的都不简单

### TSF-Linear

： Informer、Pyraformer、Autoformer、Fedformer 

提出了 6 个质疑：

1. 难道不是历史回溯窗口越长越好吗？(还真不是，输入序列太长了以后，Transformer 系的模型性能反而下降了)
2. Transformer 系模型从回溯窗口学到了什么？(close input&far input)
3. 自注意力机制有用吗？（Informer 逐渐演变为线性模型性能 upup）
4. Temporal Embedding真的保留了时间的顺序信息吗？(作者对输入颠倒顺序，发现Transformer 系的模型并没有受到太大的影响)
5. 嵌入策略起了什么作用？
6. 是训练数据集的规模限制了模型的性能吗？(然而并没有，反而保存了完整周期信息的训练数据更好)
7. 效率和复杂度的优先级真的那么重要吗？

### WITRAN

<u>RNN 系文章</u> 

水波纹信息传输循环加速网络

- 非逐点语义信息捕获
- 内存占用 &  时间复杂度

贡献：

（1）水波纹信息传输：WIT

（2）水平垂直 门控选择单元： HVGSU

（3）循环加速网络：RAN
