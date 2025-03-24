# Fourier级数

- [x] 傅里叶级数
- [x] 傅里叶变换

- [ ] 离散傅里叶变换
- [ ] 快速傅里叶变换

## 三角函数形式

🟢  case1：周期= $2 \pi$

> 周期为 $2\pi$的函数展开式

$$f(t) = \frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n cosnt + b_nsinnt$$

$$\left\{
\begin{aligned}
a_0 & =\frac{1}{\pi}\int_{-\pi}^{\pi}f(t)\mathrm{d}t, \\
a_n & =\frac{1}{\pi}\int_{-\pi}^{\pi}f(t)\cos\mathrm{n}t\mathrm{d}t, \\
b_n & =\frac{1}{\pi}\int_{-\pi}^{\pi}f(t)\sin\mathrm{n}t\mathrm{d}t
\end{aligned}\right.$$

🟢 case2：周期=2T

$t=?x = 2\pi \frac{x}{2T} = \pi \frac{x}{T}$

$$f(x)=\frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n cosn\pi\frac{x}{T} + b_nsinn\pi\frac{x}{T}$$

> $dt = \frac{\pi}{T}dx$
>
> $t \in (-\pi,\pi)  → x \in (-T,T)$

$$\left\{
\begin{aligned}
a_0 & =\frac{1}{T}\int_{-T}^{T}f(x)\mathrm{d}x, \\
a_n & =\frac{1}{T}\int_{-T}^{T}f(x)\cos\mathrm{n}\pi \frac{x}{T}\mathrm{d}x, \\
b_n & =\frac{1}{T}\int_{-T}^{T}f(x)\sin\mathrm{n}\pi \frac{x}{T}\mathrm{d}x
\end{aligned}\right.$$

🟢 case3：周期=T

$t=?x = 2\pi \frac{x}{T} =\omega x$    (令 $\omega = \frac{2\pi}{T} $)

> $dt = \frac{2\pi}{T}dx$
>
> $t \in (-\pi,\pi)  → x \in (-\frac{T}{2},\frac{T}{2})$

$f(x)=\frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n cos n2\pi\frac{x}{T} + b_nsin n2\pi\frac{x}{T}$ 

==更常用的形式：==

$$f(x)=\frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n cos n \omega x + b_nsin n \omega x$$ 

$$\left\{
\begin{aligned}
a_0 & =\frac{2}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)\mathrm{d}x, \\
a_n & =\frac{2}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)\cos\mathrm{n}\omega x\mathrm{d}x, \\
b_n & =\frac{2}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)\sin\mathrm{n}\omega x\mathrm{d}x
\end{aligned}\right.$$

特别的，$\omega_0=\frac{2\pi}{T}$ 记作基频，周期最长，最低频

## 指数形式

$$
\left\{
\begin{aligned}
e^{ix}=cosx+isinx, \\
e^{-ix}=cosx-isinx, \\
\end{aligned}\right.
$$


$$
\left\{
\begin{aligned}
cosn\omega x = \frac{e^{in\omega x}+e^{-in\omega x}}{2}, \\
i sin n\omega x = \frac{e^{in\omega x}-e^{-in\omega x}}{2}, \\
\end{aligned}\right.
$$


$f(x)=\frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n cos n \omega x + b_nsin n \omega x$

$\quad =\frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n \frac{e^{in\omega x}+e^{-in\omega x}}{2} - i b_n\frac{e^{in\omega x}-e^{-in\omega x}}{2}$

$\quad =\frac{a_0}{2} + \sum_{n=1}^{+ \infty} \frac{a_n - i b_n}{2} e^{in\omega x}+ \frac{a_n + i b_n}{2} e^{-in\omega x}$

记系数分别为：

$c_0 = \frac{a_0}{2} = \frac{1}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)\mathrm{d}x \quad (n=0)$  

$c_n = \frac{a_n - i b_n}{2} = \frac{\frac{2}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)\cos\mathrm{n}\omega x\mathrm{d}x-i\frac{2}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)\sin\mathrm{n}\omega x\mathrm{d}x}{2}$

$ \quad = \frac{\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)\cos\mathrm{n}\omega x\mathrm{d}x-if(x)\sin\mathrm{n}\omega x\mathrm{d}x}{T}$

$ \quad = \frac{\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)(\cos\mathrm{n}\omega x-i\sin\mathrm{n}\omega x)\mathrm{d}x}{T}$

$ \quad = \frac{\int_{-\frac{T}{2}}^{\frac{T}{2}} f(x) e^ {- \mathrm{n}\omega x} \mathrm{d}x}{T}$

$ \quad = \frac{1}{T} \int_{-\frac{T}{2}}^{\frac{T}{2}} f(x) e^{- \mathrm{n}\omega x }\mathrm{d} x \quad (n=1,2,3,4... ...)$

$c_{-n} = \frac{a_n + i b_n}{2} = \frac{\frac{2}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)\cos\mathrm{n}\omega x\mathrm{d}x + i\frac{2}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x)\sin\mathrm{n}\omega x\mathrm{d}x}{2}$

$ \quad = \frac{1}{T} \int_{-\frac{T}{2}}^{\frac{T}{2}} f(x) e^{\mathrm{n}\omega x }\mathrm{d} x \quad (n=1,2,3,4... ...)$

汇总 $c_0,c_n,c_{-n}$ 

可得：

$c_n = \frac{1}{T} \int_{-\frac{T}{2}}^{\frac{T}{2}} f(x) e^{- \mathrm{n}\omega x }\mathrm{d} x \quad (n=0,\pm 1,\pm 2,\pm 3,\pm 4... ...)$

最终得到，傅里叶级数的指数形式：

$$f(x)=\sum_{n= - \infty}^{\infty} c_n e^{i n \omega x }$$

$$c_n = \frac{1}{T} \int_{-\frac{T}{2}}^{\frac{T}{2}} f(x) e^{- \mathrm{n}\omega x }\mathrm{d} x \quad (n=0,\pm 1,\pm 2,\pm 3,\pm 4... ...)$$

## 物理意义

基于以上理解，扩充一些更细致的理解

![image-20250324210344659](images/image-20250324210344659.png)

==cos 形式== 可以只用 正弦函数 或者余弦函数

![image-20250324210440924](images/image-20250324210440924.png)    

![image-20250324210414287](images/image-20250324210414287.png) 

==指数形式==

==指数形式==  指数形式也证明过了

![image-20250324210541603](images/image-20250324210541603.png)

==系数之间的关系：== 

![image-20250324210750659](images/image-20250324210750659.png)

![image-20250324210800957](images/image-20250324210800957.png)

代入来代入去可得，（以 $a_n $ 和 $b_n$ 为基准）：

![image-20250324210658880](images/image-20250324210658880.png)

## 从傅里叶级数 →傅里叶变换

- 傅里叶级数是对周期为T的确定性信号做展开，而傅里叶变换将周期推广到无穷，能对具有任意长度的信号做展开
- **连续周期信号 ---> 傅里叶级数**；**连续非周期信号 ---> 傅里叶变换。**

![image-20250324220709461](images/image-20250324220709461.png) 

![image-20250324220737749](images/image-20250324220737749.png)



==最最核心的：==   

![image-20250324220853682](images/image-20250324220853682.png)

## Add

最常用的傅里叶级数形式：



![image-20250324211918264](images/image-20250324211918264.png)

![image-20250324212207779](images/image-20250324212207779.png) 

![image-20250324211939821](images/image-20250324211939821.png)  

![image-20250324211949485](images/image-20250324211949485.png)  

![image-20250324211959224](images/image-20250324211959224.png)

### 傅里叶级数推导

从无到有

![image-20250324212434771](images/image-20250324212434771.png) 

![image-20250324212502291](images/image-20250324212502291.png)

==基频&谐波== 

![image-20250324212520580](images/image-20250324212520580.png)

![image-20250324212535221](images/image-20250324212535221.png) 

![image-20250324212618891](images/image-20250324212618891.png) 

![image-20250324212631970](images/image-20250324212631970.png) 

==由三角函数推导指数形式：==

再看一遍三角函数：

![image-20250324212908554](images/image-20250324212908554.png) 

==欧拉公式==

![image-20250324212926614](images/image-20250324212926614.png) 

![image-20250324212938428](images/image-20250324212938428.png) 

![image-20250324212948994](images/image-20250324212948994.png)

![image-20250324213236544](images/image-20250324213236544.png)

==全部散开的 指数型 傅里叶级数==   

![image-20250324213154236](images/image-20250324213154236.png) 





---

reference

[傅里叶级数公式推导](https://zhuanlan.zhihu.com/p/5721100917)

[数值分析-最佳三角逼近&Fourier变换&三角插值](https://zhuanlan.zhihu.com/p/5721100917)

[(很多漂亮的图)如何理解傅立叶级数公式？](https://www.zhihu.com/question/19714540/answer/334686351)

[傅里叶系列（一）傅里叶级数的推导](https://zhuanlan.zhihu.com/p/5721100917)

[https://www.tup.com.cn/upload/books/yz/079859-01.pdf](https://www.tup.com.cn/upload/books/yz/079859-01.pdf)

[纯干货数学推导 傅里叶级数与傅里叶变换 Part5\_从傅里叶级数推导傅里叶变换](https://www.bilibili.com/video/BV1jt411U7Bp?spm_id_from=333.788.videopod.sections&vd_source=99384b2db3076f120ee34966bff31479)









