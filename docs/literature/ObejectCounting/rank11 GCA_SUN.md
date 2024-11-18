![image-20241118202309048](images/image-20241118202309048.png)

2024年的新文章，可以看看

源码未公开，简单看看

![image-20241118202441321](images/image-20241118202441321.png)

arxiv日期：2024年9月18日

一眼标题：buff叠满了，SwinTransformer & Unet

作者：

18 Sep 2024 · [Yuzhe Wu](https://paperswithcode.com/author/yuzhe-wu), [Yipeng Xu](https://paperswithcode.com/author/yipeng-xu), [Tianyu Xu](https://paperswithcode.com/author/tianyu-xu), [Jialu Zhang](https://paperswithcode.com/author/jialu-zhang), [Jianfeng Ren](https://paperswithcode.com/author/jianfeng-ren), [Xudong Jiang](https://paperswithcode.com/author/xudong-jiang) 

宁波诺丁汉大学计算机科学学院

宁波诺丁汉大学卓越研究创新中心

新加坡南洋理工大学电气与电子工程学院

- [ ] 正式发表日期：

GCA 让我想起来 级联注意力

- [ ] 为什么2024年的文章，排名还这么低？
- [ ] 期刊？

[原文链接](https://arxiv.org/pdf/2409.12249v1)

源码未公开

-----

标题：GCA-SUN: A Gated Context-Aware Swin-UNet for Exemplar-Free Counting 

门控上下文感知、Swin-UNet架构、Exemplar-Free Counting（就是0-shot问题）

## 摘要

> （本文主题：Exemplar-Free Counting）Exemplar-Free Counting aims to count objects of interest without intensive annotations of objects or exemplars. 

本文的第一个提出：为了实现 Exemplar-Free Counting，提出 Gated Context-Aware Swin-UNet (GCA-SUN) 

> To achieve this, we propose Gated Context-Aware Swin-UNet (GCA-SUN) to **directly map an input image to the density map of countable objects.** 为了实现这一目标，我们提出了门控上下文感知Swin - UNet ( GCA-SUN )，将输入图像直接映射为可数物体的密度图

Swin-UNet 的功能 一句话说明：直接将输入图像映射到密度图

展开说说

> Specifically, a Gated Context-Aware Modulation module is designed in the encoder to suppress irrelevant objects or background through a gate mechanism and exploit the attentive support of objects of interest through a self-similarity matrix.
>
> 在编码器中设计了一个门控上下文感知调制模块，通过门机制来抑制无关对象或背景，并通过自相似矩阵来利用感兴趣对象的注意力支持。
>
> The gate strategy is also incorporated into the bottleneck network and the decoder to highlight the features most relevant to objects of interest.门策略也被纳入到瓶颈网络和解码器中，以突出与感兴趣对象最相关的特征。
>
> By explicitly exploiting the attentive support among countable objects and eliminating irrelevant features through the gate mechanisms, the proposed GCA-SUN focuses on and counts objects of interest without relying on predefined categories or exemplars.
>
> 通过显式地利用可数对象之间的注意力支持和通过门机制消除无关特征，GCA - SUN在不依赖预定义类别或示例的情况下关注和计数感兴趣的对象。

结果）

>  Experimental results on the FSC-147 and CARPK datasets demonstrate that GCA-SUN outperforms state-of-the-art methods. 

所用数据集：FSC147、CARPK

Index Terms—Object counting, Exemplar-free counting, Gate mechanism, Self-similarity matrix

## 引入—贡献





## 结论





## 引入







## 相关工作



