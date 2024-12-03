# GAN

[视频链接](https://www.bilibili.com/video/BV1VT4y1e796?spm_id_from=333.788.videopod.sections&vd_source=ddd7d236ab3e9b123c4086c415f4939e)

![image-20241202215829776](images/image-20241202215829776.png)

文生图模型

![image-20241202215859877](images/image-20241202215859877.png)

交互式的demo

text2image的模型 或者叫 caption2image：可以怎么构造这样一个模型呢？

前提：算力够、数据够，有大量的图像文本对

文本输入到bert中，提取文本特征，通过Transformer模型生成图像patch，然后把patch拼起来构成一张图片，假设采用 这样的模型，LOSS该怎么设计？

最常用的loss，比如L1 loss，L2 loss，归一化到0~1之间，将预测的图像像素点值跟真实的target像素点值作差，用差的绝对值 或者平方作为loss
