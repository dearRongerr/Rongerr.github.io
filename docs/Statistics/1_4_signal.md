---
hide:
  # - navigation # 显示右
  # - toc #显示左
  - footer
  - feedback
# comments: false
---

# FS、FT、DTFS、DTFT

参看课件

<div class="card">
  <div class="file-block">
    <div class="file-icon">
      <img src="/Rongerr.github.io/assets/images/icons/pdf.svg" alt="PDF Icon">
    </div>
    <div class="file-body">
      <div class="file-title">连续时间周期信号的傅里叶级数</div>
    </div>
  </div>
  <a class="down-button" target="_blank" href="/Rongerr.github.io/pdf_files/5_1_FS.pdf" markdown="1">查看</a>
</div>


<div class="card">
  <div class="file-block">
    <div class="file-icon">
      <img src="/Rongerr.github.io/assets/images/icons/pdf.svg" alt="PDF Icon">
    </div>
    <div class="file-body">
      <div class="file-title">连续时间非周期信号的傅里叶变换</div>
    </div>
  </div>
  <a class="down-button" target="_blank" href="/Rongerr.github.io/pdf_files/5_2_FT.pdf" markdown="1">查看</a>
</div>


<div class="card">
  <div class="file-block">
    <div class="file-icon">
      <img src="/Rongerr.github.io/assets/images/icons/pdf.svg" alt="PDF Icon">
    </div>
    <div class="file-body">
      <div class="file-title">离散时间周期信号的傅里叶变换</div>
    </div>
  </div>
  <a class="down-button" target="_blank" href="/Rongerr.github.io/pdf_files/5_3_DTFS.pdf" markdown="1">查看</a>
</div>


<div class="card">
  <div class="file-block">
    <div class="file-icon">
      <img src="/Rongerr.github.io/assets/images/icons/pdf.svg" alt="PDF Icon">
    </div>
    <div class="file-body">
      <div class="file-title">离散时间非周期信号的傅里叶变换</div>
    </div>
  </div>
  <a class="down-button" target="_blank" href="/Rongerr.github.io/pdf_files/5_4_DTFT.pdf" markdown="1">查看</a>
</div>


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

$ F(\omega) = \int_{-\infty}^{\infty} x(t) e^{-j\omega t}dt  $

==DTFS==  离散时间周期信号的傅里叶级数

$$x[n] = \sum_{k=0}^{N_0 - 1} F(k\Omega_0)e^{jk\Omega_0 n}$$ 

$ F(k\Omega_0)= \frac{1}{N_0}\sum_{n=0}^{N_0 - 1}x[n]e^{-jk\Omega_0 n}$

==DTFT== 离散时间周期信号的傅里叶变换 $\rightarrow FFT$

$x[n] = \frac{1}{2\pi}\int_0^{2\pi}F(\Omega)e^{j\Omega n} d\Omega$ 

$F(\Omega) = \sum_{n=-\infty}^{\infty}x[n]e^{-j\Omega n}$



<iframe src="/Rongerr.github.io/pdf_files/5_3_DTFS.pdf" width="100%" height="800px" style="border: 1px solid #ccc; overflow: auto;"> </iframe>
