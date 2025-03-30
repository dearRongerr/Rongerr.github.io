# LaTex

latex导入包

```latex
\usepackage{caption}
\usepackage{amsmath}
\usepackage{amssymb}
```

- caption
- 对齐
- \mathbb

## 公式对齐一个编号

```latex
\begin{align}
\begin{split}
\omega&=\sum\limits_{i=1}^{m}(\hat{\alpha_i}-\alpha)x_i\\
0&=\sum\limits_{i=1}^{m}(\hat{\alpha_i}-\alpha_i) \\
C&= \alpha_i + \mu_i \\
C &= \hat{\alpha}_i + \hat{\mu}_i 
\end{split}
\end{align}
```



## 多行公式对齐

![image-20231202175946443](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202503301722512.png)

## 罗马字母

$\var$

$\varepsilon$

$\epsilon$

$\Phi$

$\phi$

$\lambda_{j}\neq0$

$\infty$

$\psi$

加横线 $\overline{\psi_{j}}$

$x^{\prime}$

$\kappa$

正负号 $\pm$

$\tilde{c}$波浪线

## 偏导

在LaTeX中输入偏导数，你可以使用`\frac`命令和`\partial`符号。下面是一个简单的示例，演示如何输入一个关于变量x的偏导数：

```latex
\documentclass{article}
\begin{document}

偏导数示例：$\frac{\partial f}{\partial x}$

\end{document}
```

在这个例子中，`\frac{\partial f}{\partial x}`表示函数f关于变量x的偏导数。你可以根据需要修改变量和函数名。

## 公示编号及引用

这是一个带有编号的公式：

```latex
\begin{equation}

\label{eq:emc}
    E=mc^2
\end{equation}
```

在文本中引用公式 \ref{eq:emc}，这是爱因斯坦的质能方程。

## 定义

```latex
\begin{definition}

（熵）分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点（node）和有向边（directed node）组成。结点有两种类型：内部结点（internal node）和叶结点（leaf node）。内部结点表示一个特征或属性，叶结点表示一个类。
\end{definition}
```



## 插入图片

```latex
\begin{center}
    \includegraphics[width=0.5\textwidth]{figure/决策树模型1.png}
    \captionof{figure}{决策树模型}
    \label{fig:example}
\end{center}
```

如图`\ref{fig:label_name}`所示

```latex
\begin{center}
    \includegraphics[width=0.8\linewidth]{figure/image3.png}
    \captionof{figure}{支持向量回归的几何抽象} 
    \label{fig:支持向量回归的几何抽象}
\end{center}
```

```latex
\begin{center}
	\includegraphics[width=0.8\textwidth]{figure/E1.png}
\end{center}
```

```latex
\begin{center}
    \includegraphics[width=1\linewidth]{figure/image2.png}
    \captionof{figure}{支持向量回归}  
\end{center}
```



## 当前位置三线表

```latex
\begin{table}[h] % 使用 [h] 选项将表格放置在当前位置
  \centering
  \caption{历史及实时路况信息}
  \begin{tabular}{ccc}
    \hline
    字段名称 & 字段含义 \\
    \hline
    link & 数据2  \\
    label & 数据5 \\
    current_slice_id & 数据5 \\
    future_slice_id & 数据5 \\
    recent_feature & 数据5 \\
    history_feature & 数据5 \\
    \hline
  \end{tabular}
\end{table}
```



```latex
在文本中插入一个三线表如下所示：

\begin{center}
  \captionof{table}{这是一个三线表的例子}
  \begin{tabular}{ccc}
    \toprule
    列1 & 列2 & 列3 \\
    \midrule
    数据1 & 数据2 & 数据3 \\
    数据4 & 数据5 & 数据6 \\
    \bottomrule
  \end{tabular}
\end{center}

在表格之后的文本继续。
```



```latex
\begin{table}[h]
	\caption{数据集示例}	
	\centering
	\begin{tabular}{ccc}
		\toprule
		\textbf{序号} & \textbf{Date} & \textbf{Number of Passengers}\\
		\midrule
		1 & 1949/1/1 & 112 \\
		2 & 1949/2/1 & 118 \\
            3 & 1949/3/1 & 132 \\
            4 & 1949/4/1 & 129 \\
            5 & 1949/5/1 & 121 \\
            ... & ... & ... \\
            144 & 1960/11/1 & 390 \\
            145 & 1949/12/1 & 432 \\
		\bottomrule
	\end{tabular}
\end{table}
```



## latex 单元格内换行

```latex
\begin{table}
  \centering
  \caption{三线表段内分行}
  \begin{tabular}{ccc}
    \toprule
    列1 & 列2 & 列3 \\
    \midrule
    数据1 & 数据2 & \begin{tabular}[t]{@{}c@{}}数据3 \\ 行2\end{tabular} \\
    数据4 & \begin{tabular}[t]{@{}c@{}}数据5 \\ 行1\end{tabular} & 数据6 \\
    \bottomrule
  \end{tabular}
\end{table}
```

## 计数列表

```latex
\begin{enumerate}
	\item 决策树调参剪枝实现手写数字识别.
	\item 随机森林实现手写数字识别.
        \item 补充实验.
\end{enumerate}
```

## 不计数列表

```latex
\begin{itemize}
	\item 
\end{itemize}

# 
```

## 表格模版

```latex
\begin{table}
	\caption{这是一个三线表.}	
	\centering
	\begin{tabular}{ccc}
		\toprule
		\textbf{Treatments} & \textbf{Response 1} & \textbf{Response 2}\\
		\midrule
		Treatment 1 & 0.0003262 & 0.562 \\
		Treatment 2 & 0.0015681 & 0.910 \\
		\bottomrule
	\end{tabular}
\end{table}
```



## 代码

```latex
\begin{lstlisting}[caption = cs代码表测试]


\end{lstlisting}

\begin{lstlisting}


\end{lstlisting}
```



## 求和号正上正下

```latex
$\sum\limits_{n=0}^{\infty}2^{n} = -1$
```

$\sum\limits_{n=0}^{\infty}2^{n} = -1$



## 代码 等宽字体

在LaTeX中，你可以使用`\texttt{}`命令来显示等宽字体文本，以突出代码或库的名称。对于"Scikit-learn库`\texttt{Scikit-learn}`库

## 学术型论文常用的定理类环境的定义

```latex
%===================  定理类环境定义 ===================
\newtheorem{example}{例}              % 整体编号
\newtheorem{algorithm}{算法}
\newtheorem{theorem}{定理}[section]            % 按 section 编号
\newtheorem{definition}[theorem]{定义}
\newtheorem{axiom}[theorem]{公理}
\newtheorem{property}[theorem]{性质}
\newtheorem{proposition}[theorem]{命题}
\newtheorem{lemma}[theorem]{引理}
\newtheorem{corollary}[theorem]{推论}
\newtheorem{remark}[theorem]{注解}
\newtheorem{condition}[theorem]{条件}
\newtheorem{conclusion}[theorem]{结论}
\newtheorem{assumption}[theorem]{假设}
\usepackage{amsmath}
\numberwithin{equation}{section} % 按 section 编号
```

## 算法格式引用

```latex
\begin{algorithm}

\label{alg:example}
\end{algorithm}

引用算法：算法~\ref{alg:example} 展示了一个简单的示例算法。

#插入链接
\href{https://example.com}{bagging nearest neighbor classifiers}
```

## 引用参考文献

```latex
\cite
```

## latex 正体、花体

花体字母要调包

```latex
\usepackage{amsthm,amsmath,amssymb}
\usepackage{mathrsfs}
```

`\mathrm{R}`$\mathrm{R}$

`\rm{R}` $\rm{R}$

`\mathbb{R}` $\mathbb{R}$

`\mathcal{R}` $\mathcal{R}$

`\mathscr{R}` $\mathscr{R}$

## latex 字母的正上方打两个点

$\ddot x$

`\ddot x` 

## 定义

```latex
\begin{definition}{Oracle方法}

\label{def:example}

​    这是一个定义。
\end{definition}

在定义~\ref{def:example} 中我们可以看到...
```

## 引用

- 图片
- 公式
- 参考文献
- 附录
- 表格
- 定义
- 定理

```latex
\begin{theorem}

\label{thm:example}    

这是一个定理。

 \end{theorem} 

在定理~\ref{thm:example} 中我们可以看到...
```

## 打波浪线

$\widetilde{{\beta}}=A\boldsymbol{y}$

```latex
\widetilde{{\beta}}=A\boldsymbol{y}
```

## 小于等于大于等于

```latex
\begin{aligned}1\leqslant j\leqslant p\end{aligned}
```

$\begin{aligned}1\leqslant j\leqslant p\end{aligned}$

## 大帽子

$\widehat{\mathcal{M}_d}$

```latex
$\widehat{\mathcal{M}_d}$
```

## 公式水平对齐 去掉公式编号

```latex
\begin{flalign}
&\ 公式内容1  &
&\ 公式内容2  &
&\ 公式内容3  &
\nonumber
\end{flalign}
```


$$
\begin{flalign}
&\ 公式内容1  &
&\ 公式内容2  &
&\ 公式内容3  &
\nonumber
\end{flalign}
$$

## 打括号

![image-20231201211943483](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202503301708876.png)

![image-20231201212007641](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202503301708524.png)



## 字体加颜色

```latex
首先，在最前面的导言区输入命令 **\usepackage{color}**，然后，在需要加注颜色的处输入**{\color{red}{I love you}}**，最外面的大括号是只对“I love you”加注红色，如果没有这个大括号的话，那会对后面所有的文本加颜色。如果要对公式加注颜色，只需要将文字“I love you”换成公式的命令就可以了，如：要对公式*A*（*x*）加颜色，输入{\color{red}{$*A*（*x*）$}}就可以了。
```

## 三线表

```latex
\begin{table}
	\caption{这是一个三线表.}	
	\centering
	\begin{tabular}{ccc}
		\toprule
		\textbf{Treatments} & \textbf{Response 1} & \textbf{Response 2}\\
		\midrule
		Treatment 1 & 0.0003262 & 0.562 \\
		Treatment 2 & 0.0015681 & 0.910 \\
		\bottomrule
	\end{tabular}
\end{table}

```



![image-20250330171103699](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202503301711868.png)

```latex
\begin{table}[!htp]
	\centering
	% PLCR已经定义
	\caption{某校学生身高体重样本.}
	\label{tab2:heightweight}	
	\begin{tabular}{lccc}
		\toprule
		序号&年龄&身高&体重\\
		\midrule
		1&14&156&42\\
		2&16&158&45\\
		3&14&162&48\\
		4&15&163&50\\
		\cmidrule{2-4}
		平均&15&159.75&46.25\\
		\bottomrule
	\end{tabular}
\end{table}
```



![image-20250330171141749](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202503301711911.png)



## latex空 2 行

\vspace{2ex}

## latex beamer 行间距

\linespread{2}

## latex 多行公式

```latex
\begin{equation}
\left\{
             \begin{array}{lr}
             x=\dfrac{3\pi}{2}(1+2t)\cos(\dfrac{3\pi}{2}(1+2t)), &  \\
             y=s, & 0\leq s\leq L,|t|\leq1.\\
             z=\dfrac{3\pi}{2}(1+2t)\sin(\dfrac{3\pi}{2}(1+2t)), &  
             \end{array}
\right.
\end{equation}
```



```latex
\begin{equation}
\begin{aligned}
\begin{split}
& \min_{ \omega,b} \quad\frac{1}{2}|| \omega||^2 + C \sum \limits_{i=1}^m(\xi_i + \hat\xi_i)\\
& st.  \left\{ \begin{array}{ll} f(x_i)- y_i \leq \epsilon + \xi_i \\ y_i-f(x_i)\leq \epsilon + \hat{\xi}_i \\ \xi_i >0,\hat{\xi}_i>0 \end{array}  \quad \quad i=1,2,3,...,m \right.
\end{split}
\end{aligned}   
\end{equation}


\begin{equation}
\begin{aligned}
\begin{split}

\end{split}
\end{aligned}   
\end{equation}
```

## beamer 一点一点的出来的动画

```latex
\subsection{作者在这篇综述中具体讲了什么？ }
\begin{frame}{Frame Title}
    The main issues and contributions of this paper are as follows:
\begin{itemize}[<+->]
\item Contribution 1: ...... ;
\item Contribution 2: ...... ;
\item Contribution 3: ...... ;
\end{itemize}  
\end{frame}
```

## beamer 算法跨页 

```latex
\begin{frame}[allowframebreaks]
```



```latex
\documentclass{beamer}
\usefonttheme{serif}
\usetheme{Warsaw}
%%=====================================================================================
%算法宏包
\usepackage{algorithm,algpseudocode}
\makeatletter
\newenvironment{breakablealgorithm}
{
		\begin{center}
			\refstepcounter{algorithm}% New algorithm
			\hrule height.8pt depth0pt \kern2pt% \@fs@pre for \@fs@ruled
			\renewcommand{\caption}[2][\relax]{% Make a new \caption
				%{\raggedright\textbf{\textbf{算法}~\thealgorithm} ##2\par}%
               {\raggedright\textbf{\ALG@name~\thealgorithm} ##2\par}%
				\ifx\relax##1\relax % #1 is \relax
				\addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##2}%
				\else % #1 is not \relax
				\addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##1}%
				\fi
				\kern2pt\hrule\kern2pt
			}
		}{
		\kern2pt\hrule\relax %\@fs@post% for \@fs@ruled
	\end{center}
}
\makeatother
%===========================================================================================
\newcommand{\rr}{\tilde{r}_{0}}
\begin{document}
%%=====================================================
\begin{frame}[allowframebreaks]
\begin{breakablealgorithm}
\caption{Your Algorithm}%算法标题
	\begin{algorithmic}[1]%一行一个标行号
     ******************************
     ******************************
     ******************************
%%*为算法具体内容
	\end{algorithmic}
\end{breakablealgorithm}
\end{frame}

\end{document}
```



## beamer 16:9

```latex
\documentclass[aspectratio=169]{beamer}
```

## 行间距

```latex
\linespread{1.5}%修改行距
```

## 最后一页致谢 大写 在中间

```latex
\begin{frame}
\Huge{\centerline{TheEnd}}
\end{frame}
```

