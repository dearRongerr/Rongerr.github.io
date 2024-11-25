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

Our contributions can be summarized as follows. 

1. The proposed GCA-SUN achieves exemplar-free counting through a UNet-like architecture that utilizes Swin transformer blocks for feature encoding and decoding, avoiding the sample bias of exemplar-based approaches [11].   所提出的GCA - SUN通过类UNet结构实现了无样本计数，该结构利用Swin变换块进行特征编码和解码，避免了样本偏差（EFC计数）
2. The proposed GCAM exploits attentive support of repetitive objects through the self similarity matrix, to focus on countable objects. 提出的GCAM通过自相似矩阵利用对重复对象的细心支持，聚焦于可数对象。
3.  The gate mechanism is integrated into various modules, e.g., GCAM, GEFS and GAFU, which suppresses the features of irrelevant objects or background while highlighting the most relevant features to countable objects.  将门机制集成到GCAM、GEFS和GAFU等模块中，在突出与可数对象最相关的特征的同时，抑制无关对象或背景的特征。
4. The proposed GCA-SUN is evaluated on the FSC-147 and CARPK datasets. It outperforms state-of-the-art methods for exemplar-free counting.  在FSC - 147和CARPK数据集上对提出的GCA - SUN进行评估。在无样本计数方面，它优于当前最先进的方法。

- 本文提出的网络结构： GCA-SUN、使用了SwinTransformer
- 模块： GCAM
- 门机制：GCAM, GEFS and GAFU
- 数据集： FSC-147 and CARPK datasets

## 结论

**一、GCA-SUN**

The proposed GCA-SUN effectively tackles the problems of exemplar-free counting by using a Swin-UNet architecture to directly map the input image to the density map of countable objects. 

GCA - SUN通过使用Swin - UNet架构将输入图像直接映射到可数物体的密度图，有效地解决了无样本计数问题。

**二、GCAM** 

The proposed GCAM exploits the attention information among the tokens of repetitive objects through the self-similarity matrix, and suppresses the features of irrelevant objects through a gate mechanism.

所提出的GCAM通过自相似矩阵挖掘重复对象标记间的注意力信息，并通过门机制抑制无关对象的特征。

**三、The gate mechanism &  GEFS module  & GAFU module**

The gate mechanism is also incorporated into the GEFS module and the GAFU module, which highlight the features most relevant to countable objects while suppressing irrelevant ones. 

门机制也被纳入到GEFS模块和GAFU模块中，突出与可数对象最相关的特征，同时抑制不相关的特征。

**四、结果**

Our experiments on the FSC-147 and CARPK datasets demonstrate that GCASUN outperforms state-of-the-art methods, achieving superior performance in both intra-domain and cross-domain scenarios.

在FSC - 147和CARPK数据集上的实验表明，GCASUN优于现有的方法，在域内和跨域场景中都取得了优异的性能。

## 引入

### 第一段 目标计数分成三类，特定类别计数、类无关计数、exemplar-free计数

Object counting determines the number of instances of a specific object class in an image [1], e.g., vehicles [2], crowd [3], and cells [4]. It can be broadly categorized as:

1) Class-Specific Counting (CSC), counting specific categories like fruits [5] and animals [6]; 
2) Class-Agnostic Counting (CAC), counting objects based on visual exemplars [1], [7], [8] or text prompts [9], [10]; 
3) Exemplar-Free Counting (EFC), counting objects without exemplars, presenting a significant challenge in discerning countable objects and determining their repetitions [8], [11], [12].

!!! note

	CSC计数、CAC计数；特定类别计数、类别不敏感计数
	
	FSC计数、ZSC计数；小样本计数、0样本计数

### 第二段   Exemplar-Free Counting (EFC)的研究现状

Exemplar-Free Counting shows promise for automated systems such as wildlife monitoring [13], healthcare [14], and anomaly detection [15]. 

Hobley and Prisacariu directly regressed the image-level features learned by attention modules into a density map [12]. 

==CounTR [8] and LOCA [16]== are originally designed for CAC tasks, but can be adapted to EFC tasks by using trainable components to simulate exemplars. 

==RepRPN-Counter i==dentifies exemplars from region proposals by majority voting [11], and ==DAVE== selects valuable objects using a strategy similar to majority voting based on [17].

### 第三段 GCA-SUN=encoder + bottleneck+decoder

现有EFC计数存在不足 RepRPN-Counter 

> Despite the advancements, existing models [8], [16], [17] often explicitly require exemplars to count similar objects.EFC methods such as RepRPN-Counter do not require exemplars but generate them through region proposal [11]. Either explicit or implicit exemplars may induce sample bias as exemplars can’t cover the sample distribution.  由于样例无法覆盖样本分布，不论是外显样例还是内隐样例都可能导致样本偏差。

门控上下文感知的 Gated Context-Aware Swin-UNet (GCA-SUN)；直接将输入图片映射成密度图，不需要任何示例

> To address the challenge, we propose Gated Context-Aware Swin-UNet (GCA-SUN), which directly maps an input image to the density map of countable objects, without any exemplars. 

encoder包含两部分：

- SwinTransformer 提取特征
- 门控感知模块 

> Specifically, the encoder consists of a set of Swin Transformers to extract features, and Gated Context-Aware Modulation (GCAM) blocks to exploit the attentive supports of countable objects. 

bottleneck network  门控增强特征提取器，增强 encoder特征 Gated Enhanced Feature Selector (GEFS)

> The bottleneck network includes a Gated Enhanced Feature Selector (GEFS) to emphasize the encoded features that are relevant to countable objects. 

decoder：

- SwinTransformer 生成密度图：结合Gated Adaptive Fusion Units (GAFUs) 门控适应融合单元，按照与目标的相关度进行加权
- 最后 回归头 被使用，从加权特征中产生密度图

> The decoder includes a set of Swin transformers for generating the density map, with the help of Gated Adaptive Fusion Units (GAFUs) to selectively weigh features based on their relevance to countable objects. Finally, a regression head is utilized to derive the density map from the aggregated features.

!!! note  "总结这段"

	1. Gated Context-Aware Swin-UNet (GCA-SUN)
	2. encoder= Swin Transformers +  Gated Context-Aware Modulation (GCAM) blocks 
	3. bottleneck = Gated Enhanced Feature Selector (GEFS)
	4. decoder = Swin transformers + Gated Adaptive Fusion Units (GAFUs)
	5. regression head

用了很多门控机制

### 第四段 GCAM

> One key challenge in EFC is to effectively differentiate countable objects from other objects. 
>

EFC的一个挑战是 如何有效的从其他目标中 区分出 计数物体；

EFC中的一个关键挑战是有效地区分可数对象与其他对象

> The GCAM blocks tackle the challenge by first evaluating feature qualities by computing the feature score for each token, and then prioritizing those with informative content.

GCAM模块首先通过计算每个token的特征得分来评估特征质量，然后优先考虑那些具有信息含量的特征。

GCAM模块可以解决这个问题

!!! note "GACM模块的功能"
	问题：effectively differentiate countable objects from other objects   
	Solution：GCAM

> In addition, GCAM computes pairwise similarities between tokens through a self-similarity matrix, exploiting the support of repeating objects in the same scene.

此外，GCAM通过自相似矩阵计算token之间的成对相似度，利用同一场景中重复对象的支持度。

> Lastly, a gate mechanism is incorporated to highlight the most relevant features while suppressing irrelevant ones.

最后，引入门机制，突出最相关的特征，同时抑制不相关的特征。‘

### 第五段

Another challenge is that foreground objects often share similar low-level features with background content. 

另一个挑战是，前景对象往往与背景内容共享相似的低级特征。

The skip connections directly fuse low-level features in the encoder with high-level semantics in the decoder, potentially impeding counting performance as the background information could disturb the foreground objects. 

跳跃连接直接将编码器中的低级特征与解码器中的高级语义进行融合，由于背景信息会对前景物体产生干扰，可能会影响计数性能。

To tackle this issue, gate mechanisms are incorporated into both GEFS and GAFU to suppress irrelevant low-level features while preserving as much information on objects of interest as possible. 

为了解决这个问题，在GEFS和GAFU中都融入了门机制，以抑制不相关的低级特征，同时尽可能多地保留感兴趣对象的信息。

The former selectively enhances the compressed features at the bottleneck, and the latter filters the features in the decoder.

前者选择性地增强瓶颈处的压缩特征，后者在解码器中过滤特征。

第六段：贡献

## 相关工作

没有相关工作，本文的目录结构：

标题：GCA-SUN: A Gated Context-Aware Swin-UNet for Exemplar-Free Counting

Abstract

**I. INTRODUCTION**

**II. PROPOSED METHOD**

> A. Overview of Proposed Method
>
> B. Swin-T Encoder with GCAM
>
> C. Bottleneck with GEFS
>
> D. Swin-T Decoder with GAFU

**III. EXPERIMENTAL RESULTS**

> A. Experimental Settings
>
> B. Comparison with State-of-the-Art Methods
>
> C. Cross-Domain Evaluation on CARPK Dataset
>
> D. Visualization of GCAM 
>
> E. Ablation Study

**IV. CONCLUSION**

!!!note "24·11·19：第一次读，回顾"

	CSC计数，几个计数的简写
	本文门控机制多
	其实，没用U-Net？SwinTransformer比较类似U-Net
