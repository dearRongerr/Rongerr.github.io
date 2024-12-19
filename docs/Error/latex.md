# Latex

- [x] overleaf全是红线

menu—speak cheak：off

- [x] `% 去掉thebibliography环境自带的“参考文献”标题`

问题描述：`.cls`文件中的声明，在文档渲染的时候，自动出现“参考文献”字样，没有编号且不在目录中编号

```
\bibliographystyle{IEEEtran}
```

解决：

```python
\renewcommand{\refname}{\section{参考文献}}
\bibliography{books}
```

- [x] section格式设置

![image-20241213145341078](images/image-20241213145341078.png)

```latex
\setcounter{page}{1}

\CTEXsetup[format={\Large\bfseries}]{section}

\begin{center}
\section*{\textbf{开 题 报 告 正 文}}
\end{center}

学位论文研究课题：

\textbf{课题来源：}1.纵向课题（  ）；2.横向课题（  ）；3.自由选题（√  ）；4.其他（  ）。请打“√”。

\section{立题依据}

\subsection{研究背景及意义}
```

![image-20241213145426732](images/image-20241213145426732.png)

