# 信号的合成与分解

> **傅里叶级数的三角形式、指数形式、相位、频谱**
>
> 傅里叶的三角函数：
>
> 正交基：$\cos1\omega x,\sin1\omega x,\cos2\omega x,\sin2\omega x,\cos3\omega x,\sin3\omega x,...,$
>
> 傅里叶的指数函数形式：
>
> 正交基：$\{e^{-jn\omega t},...,e^{-j2\omega t},e^{-j\omega t},e^{j0\omega t},e^{j\omega t},e^{j2\omega t},...e^{jn\omega t}\}$

[数字信号处理_信号的合成与分解](https://www.bilibili.com/video/BV1QG4y127B1?spm_id_from=333.788.videopod.sections&vd_source=ddd7d236ab3e9b123c4086c415f4939e)

复数形式：$a+bi$

复指数形式：$\sqrt{a^2+b^2}(cos\theta+i sin\theta)$ 

其中，

$cos\theta = \frac{a}{\sqrt{a^2+b^2}}$

$sin\theta = \frac{b}{\sqrt{a^2+b^2}} $

![image-20250325233155639](images/image-20250325233155639.png)

**向量的合成与分解** 

![image-20250326095511196](images/image-20250326095511196.png)

**函数内积**：加法变积分运算

![image-20250326095608113](images/image-20250326095608113.png)

**三角函数集的正交性**

![image-20250326095722638](images/image-20250326095722638.png)

**结论：**

- 任意两个不同函数内积=0
- 相同函数内积 $\neq 0$ 

![image-20250326095752336](images/image-20250326095752336.png)

**傅里叶级数：**

![image-20250326095853200](images/image-20250326095853200.png)

- 正交基是：

$\{1,\cos x,\sin x,\cos 2x,\sin 2x,...,\cos nx,\sin nx\}$ 

- 傅里叶级数用的是：

$\cos1\omega x,\sin1\omega x,\cos2\omega x,\sin2\omega x,\cos3\omega x,\sin3\omega x,...,$

- [ ] 为什么呢？

- 周期信号的傅里叶级数
- 对于周期信号，使用直流信号，一些列

