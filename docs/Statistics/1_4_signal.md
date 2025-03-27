# FS、FT、DTFS、DTFT

课件

[连续时间周期信号的傅里叶级数](../pdf_files/5_1_FS.pdf)

[连续时间非周期信号的傅里叶变换](../pdf_files/5_2_FT.pdf)

[离散时间周期信号的傅里叶变换](../pdf_files/5_3_DTFS.pdf)

[离散时间非周期信号的傅里叶变换](../pdf_files/5_4_DTFT.pdf)

==目录：== 

![image-20250327183609041](images/image-20250327183609041.png)

目标都是：原始信号表示为一系列单位圆上的正交基表示 $x(t)=... ...$

- 在正交基上的强度，用内积计算

==FS==   连续时间周期信号的傅里叶级数

【时域→频域】

$$x(t) = \sum_{k=-\infty}^{\infty}F(k\omega_0)e^{jk\omega_0 t}$$

其中：【频域→时域】

 $F(k\omega_0) = \frac{1}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}} x(t)e^{-jk\omega_0t}dt$

==FT==  连续时间非周期信号的傅里叶级数

$$x(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty}F(\omega)e^{j\omega t}d\omega $$ 

其中，

$ F(\omega) = \int_{-\infty}^{\infty}x(t)e^{-j\omega t}dt  $

==DTFS==  离散时间周期信号的傅里叶级数

$$x[n] = \sum_{k=0}^{N_0 - 1} F(k\Omega_0)e^{jk\Omega_0 n}$$ 

$ F(k\Omega_0)= \frac{1}{N_0}\sum_{n=0}^{N_0 - 1}x[n]e^{-jk\Omega_0 n}$

==DTFT== 离散时间周期信号的傅里叶变换 $\rightarrow FFT$

$x[n] = \frac{1}{2\pi}\int_0^{2\pi}F(\Omega)e^{j\Omega n} d\Omega$ 

$F(\Omega) = \sum_{n=-\infty}^{\infty}x[n]e^{-j\Omega n}$

## FS

![image-20250327180515334](images/image-20250327180515334.png)

吉布斯现象：

![image-20250327180543469](images/image-20250327180543469.png) 

## FT

对比FS

![image-20250327181144477](images/image-20250327181144477.png) 

![image-20250327181205102](images/image-20250327181205102.png) 

![image-20250327181217085](images/image-20250327181217085.png) 

### FS实例：方波信号

![image-20250327181247856](images/image-20250327181247856.png) 

![image-20250327181314286](images/image-20250327181314286.png) 



## DTFS

![image-20250327184320384](images/image-20250327184320384.png) 

### 实例

![image-20250327184338705](images/image-20250327184338705.png) 

![image-20250327184353205](images/image-20250327184353205.png) 

![image-20250327184406488](images/image-20250327184406488.png) 

![image-20250327184419258](images/image-20250327184419258.png) 

![image-20250327184436790](images/image-20250327184436790.png)



