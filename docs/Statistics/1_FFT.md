# FFT

🟢  case1：周期= $2 \pi$

> 周期为 $2\pi$的函数展开式

$f(t) = \frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n cosnt + b_nsinnt$

$$\left\{
\begin{aligned}
a_0 & =\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\mathrm{d}x, \\
a_n & =\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\cos\mathrm{n}x\mathrm{d}x, \\
b_n & =\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\sin\mathrm{n}x\mathrm{d}x
\end{aligned}\right.$$

🟢 case2：周期=2T

$t=?x = 2\pi \frac{x}{2T} = \pi \frac{x}{T}$

$f(x)=\frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n cosn\pi\frac{x}{T} + b_nsinn\pi\frac{x}{T}$



🟢 case3：周期=T

$t=?x = 2\pi \frac{x}{T} $

$f(x)=\frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n cos n2\pi\frac{x}{T} + b_nsin n2\pi\frac{x}{T}$ 

令 $\omega = \frac{2\pi}{T} $

$f(x)=\frac{a_0}{2} + \sum_{n=1}^{+ \infty} a_n cos n \omega x + b_nsin n \omega x$

---

case1 的系数：









