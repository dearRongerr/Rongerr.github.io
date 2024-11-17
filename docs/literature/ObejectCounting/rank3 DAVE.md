![image-20241117103916730](images/image-20241117103916730.png)

[原文链接](https://paperswithcode.com/paper/dave-a-detect-and-verify-paradigm-for-low)

[源码链接](https://github.com/jerpelhan/dave)

![image-20241117113107549](images/image-20241117113107549.png)

arxiv日期：2024年4月25日

[论文正式发表页面](https://www.computer.org/csdl/proceedings-article/cvpr/2024/530000x293/20hPgz8Mxqg)

today：241117

标题：DAVE – A Detect-and-Verify Paradigm for Low-Shot Counting 

- 两阶段计数方法：先检测再验证
- 检测：高召回，复用了LOCA的架构
- 验证：谱聚类验证

期刊：CVPR2024

引用：

```tex
@inproceedings{pelhan2024dave,
  title={DAVE-A Detect-and-Verify Paradigm for Low-Shot Counting},
  author={Pelhan, Jer and Zavrtanik, Vitjan and Kristan, Matej and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23293--23302},
  year={2024}
}
```

**LOCA  & DAVE**

![image-20241117110053231](images/image-20241117110053231.png)

## Abstract

Low-shot counters estimate the number of objects corresponding to a selected category, based on only few or no exemplars annotated in the image. The current state-ofthe-art estimates the total counts as the sum over the object location density map, but does **not provide individual object locations and sizes（提出问题）**, which are crucial for many applications. This is addressed by **detection-based counters, which, however fall behind in the total count accuracy** （基于检测的计数方法可以解决，但是准确性不高）. Furthermore, both approaches tend to overestimate the counts in the presence of other object classes due to many false positives. （这些方法的共性问题：假阳性过高）

（本文）We propose DAVE, a low-shot counter based on a detect-and-verify paradigm, that avoids the aforementioned issues by first generating a high-recall detection set and then verifying the detections to identify and remove the outliers. This jointly increases the recall and precision, leading to accurate counts. 

- **first generating a high-recall detection set and then**  
- **verifying the detections to identify and remove the outliers.** 

（结果）DAVE outperforms the top densitybased counters by ∼20% in the total count MAE, it outperforms the most recent detection-based counter by ∼20% in detection quality and sets a new state-of-the-art in zero-shot as well as text-prompt-based counting. The code and models are available on GitHub.

## Introduction-contribution

- We address the aforementioned issues by proposing a low-shot counter DAVE, which combines the benefits of density-based and detection-based formulations, and introduces a novel detect-and-verify paradigm. 

- DAVE tackles the specificity-generalization issues of the existing counters by applying a two-stage pipeline (Figure 1). 

  - **In the first**, ==detection stage== , DAVE leverages density-based estimation to obtain a high-recall set of candidate detections, which however may contain false positives. 
  - This is addressed by **the second**,  ==verification stage== , where outliers are identified and rejected by analyzing the candidate appearances, thus increasing the detection precision. Regions corresponding to the outliers are then removed from the location density map estimated in the first stage, thus improving the densitybased total count estimates as well. 

- In addition, we extend DAVE to **text-prompt-based** and to a **zero-shot** scenario, which makes DAVE the first zero-shot as well as textprompt detection-capable counter.

  ==text prompt & zero-shot==

---

**The primary contribution** of the paper is the detect-andverify paradigm for low-shot counting that simultaneously achieves high recall and precision. 

The proposed architecture is the **first** to extend to all low-shot counting scenarios. DAVE uniquely merges the benefits of both density and detection-based counting and is the **first** zero-shot-capable counter with detection output. 

（结果）

1. DAVE outperforms all stateof-the-art density-based counters on the challenging benchmark [26], including the longstanding winner [6], achieving a relative 20% MAE and 43% RMSE total-count error reductions. 
2. It also outperforms all state-of-the-art detectionbased counters on the recent benchmark FSCD147 [22] by ∼20% in detection metrics, as well as in the total count estimation by 38% MAE. 
3. Furthermore, it sets a new state-ofthe-art in text-prompt-based counting. 
4. The zero-shot DAVE variant outperforms all zero-shot density-based counters and delivers detection accuracy on-par with the most recent few-shot counters. 
5. DAVE thus simultaneously outperforms both density-based and detection-based counters in a range of counting setups.

## 5. Conclusion

**第一段**

1. We presented a novel low-shot object counting and detection method DAVE, that narrows the performance gap between density-based and detection-based counters.

DAVE是基于检测的计数方法

2.DAVE spans the entire low-shot spectrum, also covering text-prompt setups, and is the first method capable of zero-shot detection-based counting. 

   DAVE 基于文本 & 0-shot

3.This is achieved by the novel detect-and-verify paradigm, which increases the recall as well as precision of the detections.
检测&验证，准确率 召回率都很高

**第二段**

Extensive analysis demonstrates that DAVE **sets a new state-of-the-art** in total count estimation, as well as in detection accuracy on several benchmarks with comparable complexity to related methods, running 110ms/image.

In particular, DAVE **outperforms** the long-standing top low-shot counter [6], as well as the recent detection-based counter [22]. 

In a **zero-shot** setup, DAVE outperforms all density-based counters and delivers detections on par with the most recent few-shot counter that requires at least few annotations. 

DAVE also sets a new state-of-the-art in prompt-based counting.

 In our future work, we plan to explore interactive counting with the human in the loop and improve detection in extremely dense regions.未来的工作：人的交互、密集场景的检测

## 1. Introduction

P1 从Low-shot counting开始说

Low-shot counting considers estimating the number of target objects in an image, based only on a few annotated exemplars (few-shot) or even without providing the exemplars (zero-shot). Owing to the emergence of focused benchmarks [22, 26], there has been a surge in low-shot counting research recently. The current state-of-the-art low-shot counters are all density-based [6, 26, 28, 38]. This means that they estimate the total count by summing over an estimated object presence density map. Only recently, fewshot detection-based methods emerged [22] that estimate the counts as the number of detected objects.

P2 比较 Density-based & detection-based ；指出目前基于密度（Density-based ）的计数方法的不足。并说明：explainability is crucial 本文倾向于基于检测的方法

Density-based methods substantially outperform the detection-based counters in total count estimation, but they do not provide detailed outputs such as object locations and sizes. The latter are however important in many downstream tasks such as bio-medical analysis [35, 41], where explainability is crucial for human expert verification as well as for subsequent analyses. There is thus a large applicability gap between the density-based and detection-based low-shot counters.

P3  共同的缺点 假阳性过高

Furthermore, both density-based and detection-based counters are prone to failure in scenes with several object types (Figure 1). The reason lies in the **specificity  generalization tradeoff**. Obtaining a high recall requires generalizing over the potentially diverse appearances of the selected object type instances in the image. However, this also leads to false activations on objects of other categories (false positives), leading to a reduced precision and count overestimation. A possible solution is to train on multiple-class images [22], however, this typically leads to a reduced recall and underestimated counts.

P4-P5 贡献

## 2. Related Work

**第一段**

Object counting emerged as detection-based counting of objects belonging to specific classes, such as vehicles [5], cells [8], people [17], and polyps [41]. To address poor performance in densely populated regions, density-based methods [3, 4, 29–31] emerged as an alternative.

目标计数、就有检测的方法、特定类别；计数准确性，基于密度回归图的计数方法

**第二段**

All these methods rely on the availability of large datasets to train category-specific models, which, however are not available in many applications.数据集

**第三段**

Class-agnostic approaches addressed this issue by test-time adaptation to various object categories with minimal supervision.

类无关计数方法、最少的监督信号、测试阶段调整

 Early representatives [19] and [37] proposed predicting the density map by applying a siamese matching network 孪生匹配网络 to compare image and exemplar features. 比较图像&样例框特征

- Recently, the FSC147 dataset [26] was proposed to encourage the development of few-shot counting methods. Famnet [26] proposed a test-time adaptation of the backbone to improve density map estimation. FSC147数据集、 Famnet  || FSC147数据集[ 26 ]的提出鼓励了小样本计数方法的发展。Famnet [ 26 ]提出了一种骨干测试时间自适应的方法来改进密度图估计。
- BMNet+ [28] improved localization by jointly learning representation and a non-linear similarity metric. A self-attention mechanism was applied to reduce the intra-class appearance variability. BMNet + [ 28 ]通过联合学习表示和非线性相似性度量来改进定位。应用自注意力机制来减少类内外观变异性。
- SAFECount [38] introduced a feature enhancement module, improving generalization capabilities. SAFECount [ 38 ]引入了特征增强模块，提高了泛化能力。
- CounTR [16] used a vision transformer [7] for image feature extraction and a convolutional encoder to extract exemplar features. An interaction module based on cross-attention was proposed to fuse both, image and exemplar features. Coun TR [ 16 ]使用视觉转换器[ 7 ]进行图像特征提取，使用卷积编码器提取样本特征。提出了一种基于交叉注意力的交互模块来融合图像特征和样本特征。
- LOCA [6] proposed an object prototype extraction module, which combined exemplar appearance and shape with an iterative adaptation.LOCA [ 6 ]提出了一个对象原型提取模块，该模块将样本外观和形状与迭代自适应相结合。

!!! note
	**Summary   5个模型**   类别不敏感计数方法、测试阶段自适应、少量监督信号（你去看GeCo：相关工作，也是这5个模型的 研究现状

	- FamNet  
	- BMNet+  
	- SAFECount   
	- CounTR  
	- LOCA  

---

**第四段**

All few-shot counting methods require few annotated exemplars to specify the object class. With the recent development of large language models (e.g. [23]) text-prompt based counting methods emerged.

输入信号的发展：从前是标注的样例框指定类别 $\rightarrow $  现在基于文本的计数方法（大语言模型的发展）

Instead of specifying exemplars by bounding box annotations, these methods use text descriptions of the target object class. 

==❤️这段的文献概述：关于的是使用文本描述指定目标类别== 

**ZeroCLIP [36]** proposed text-based construction of prototypes, which are used to select relevant image patches acting as exemplars for counting.

ZeroCLIP [ 36 ]提出了基于文本的原型构造，用于选择相关的图像块作为计数的范例。

**CLIPCount [15]** leveraged CLIP [23] for image-text alignment and introduced patch-text contrastive loss for learning the visual representations used for density prediction. 

CLIPCount [ 15 ]利用CLIP [ 23 ]进行图像-文本对齐，并引入块-文本对比损失来学习用于密度预测的视觉表示。

Several works [13, 25] address the extreme case in which no exemplars are provided and the task is to count the majority class objects (i.e., zero-shot counting).

一些工作[ 13、25]解决了没有提供样例的极端情况，其任务是对多数类对象进行计数(即零样本计数)
!!! note
	summary   
	1. ZeroCLIP   
	2. CLIPCount   

!!! info
	DAVE   
	①基于检测    
	②text-prompt   
	③zero-shot

---

**第五段**

With minimal architectural changes, the recent few-shot methods [6, 16] also demonstrated a remarkable zero-shot counting performance. A common drawback of densitybased counters is that they do not provide object locations.

在结构变化很小的情况下，最近的少样本方法[ 6、16 ]也表现出了出色的零样本计数性能。基于密度的计数器的一个共同缺点是它们不提供对象位置。



![image-20241117140134150](images/image-20241117140134150.png)

![image-20241117140210571](images/image-20241117140210571.png)



[6]  LOCA  [16]CounTR

开始引出可以提供定位的计数 方法

**第六段**

To address the aforementioned limitation of density based counters, the first few shot counting and detection method [22] has been recently proposed by extending a transformer-based object detector [2] with an ability to detect objects specified by exemplars.

<u>文献22 第一个基于检测的FSC方法</u> 

![image-20241117135756279](images/image-20241117135756279.png)

 However, the detection based counter falls far behind in total count estimation compared with the best density-based counters.

基于检测的方法，最大的弊端：计数准确性不高(241117)

## 3. Counting by detection and verification

todo