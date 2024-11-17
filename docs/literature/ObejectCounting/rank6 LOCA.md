![image-20241117155038213](images/image-20241117155038213.png)

[原文链接](https://paperswithcode.com/paper/a-low-shot-object-counting-network-with)

[源码链接](https://github.com/djukicn/loca)

引用：

```
@InProceedings{Dukic_2023_ICCV,
    author    = {{\DJ}uki\'c, Nikola and Luke\v{z}i\v{c}, Alan and Zavrtanik, Vitjan and Kristan, Matej},
    title     = {A Low-Shot Object Counting Network With Iterative Prototype Adaptation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {18872-18881}
}
```

arxiv日期：2023年9月28日

[正式发表页面](https://openaccess.thecvf.com/content/ICCV2023/papers/Dukic_A_Low-Shot_Object_Counting_Network_With_Iterative_Prototype_Adaptation_ICCV_2023_paper.pdf) ICCV2023

today 241117

![image-20241117155355072](images/image-20241117155355072.png)

GeCo作者：27 Sep 2024 · [Jer Pelhan](https://paperswithcode.com/author/jer-pelhan), [Alan Lukežič](https://paperswithcode.com/author/alan-lukezic-1), [Vitjan Zavrtanik](https://paperswithcode.com/author/vitjan-zavrtanik), [Matej Kristan](https://paperswithcode.com/author/matej-kristan)

LOCA作者： [ICCV 2023 ](https://paperswithcode.com/conference/iccv-2023-1) · [Nikola Djukic](https://paperswithcode.com/author/nikola-djukic), [Alan Lukezic](https://paperswithcode.com/author/alan-lukezic), [Vitjan Zavrtanik](https://paperswithcode.com/author/vitjan-zavrtanik), [Matej Kristan](https://paperswithcode.com/author/matej-kristan) 

DAVE作者：25 Apr 2024 · [Jer Pelhan](https://paperswithcode.com/author/jer-pelhan), [Alan Lukežič](https://paperswithcode.com/author/alan-lukezic-1), [Vitjan Zavrtanik](https://paperswithcode.com/author/vitjan-zavrtanik), [Matej Kristan](https://paperswithcode.com/author/matej-kristan) 

CounTR作者：29 Aug 2022 · [Chang Liu](https://paperswithcode.com/author/chang-liu), [Yujie Zhong](https://paperswithcode.com/author/yujie-zhong), [Andrew Zisserman](https://paperswithcode.com/author/andrew-zisserman), [Weidi Xie](https://paperswithcode.com/author/weidi-xie)  SHJT

## 摘要

We consider low-shot counting of arbitrary semantic categories in the image using only few annotated exemplars (few-shot) or no exemplars (no-shot). 计数任意语义类别

The standard few-shot pipeline follows extraction of appearance queries from exemplars and matching them with image features to infer the object counts. 

Existing methods extract queries by feature pooling which neglects the shape information (e.g., size and aspect) and leads to a reduced object localization accuracy and count estimates. 丢失了形状信息和定位信息

We propose a **L** ow-shot  **O** bject **C** ounting network with iterative prototype **A** daptation (LOCA). 

Our main contribution is the new object prototype extraction module, which iteratively fuses the exemplar shape and appearance information with image features. 

目标原型提取模块，迭代融合示例框形状和外观信息

The module is easily adapted to zero-shot scenarios, enabling LOCA to cover the entire spectrum of low-shot counting problems. 

可以适用于0-shot场景

LOCA outperforms all recent state-of-the-art methods on FSC147 benchmark by 20-30% in RMSE on one-shot and fewshot and achieves state-of-the-art on zero-shot scenarios, while demonstrating better generalization capabilities. The code and models are available.

## 引入-贡献

We propose a Low-shot Object Counting network with iterative prototype Adaptation (LOCA). **Our main contribution** is the new object prototype extraction module, which separately extracts the exemplar shape and appearance queries. The shape queries are gradually adapted into object prototypes by considering the exemplar appearance as well as the appearance of non-annotated objects, obtaining excellent localization properties and leading to highly accurate counts (Figure 1). To the best of our knowledge, LOCA is the first low-shot counting method that explicitly uses exemplars shape information for counting. In contrast to most works [26, 24, 30, 31], LOCA does not attempt to transfer exemplar appearance onto image features, but rather constructs strong prototypes that generalize across the image-level intra-class appearance.

我们的贡献：

- 

LOCA outperforms all state-of-the-art (in many cases more complicated methods) on the recent FSC147 benchmark [24]. On the standard few-shot setup it achieves ∼30% relative performance gains, on one-shot setup even outperforms methods specifically designed for this setup, achieves state-of-the-art on zero-shot counting. In addition, LOCA demonstrates excellent cross-dataset generalization on the car counting dataset CARPK [12].

## 结论





## 引入







## 相关工作
