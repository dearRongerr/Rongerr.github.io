# 功能开发测试页面
- 可折叠的下拉框

- 卡片展示嵌入的pdf、压缩包等文件
  

???+ info

    - Author: Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, Bryan Hooi
    - Conference: ICLR 2024
    - arXiv: [2306.13063](https://arxiv.org/abs/2306.13063)

docs/assets/images/icons/pdf.svg

<div class="card file-block" markdown="1">
<div class="file-icon"><img src="/Rongerr.github.io/assets/images/icons/pdf.svg" style="height: 3em;"></div>
<div class="file-body">
<div class="file-title">论文</div>
<div class="file-meta">1.15 MB / 29 P / 2025-02-25</div>
</div>
<a class="down-button" target="_blank" href="/Rongerr.github.io/pdf_files/1_0_dilatedConv.pdf" markdown="1">:fontawesome-solid-download: 下载</a>
</div>

路径：

本地 vscode 中：docs/pdf_files/1_0_dilatedConv.pdf

mkdocs serve：127.0.0.1.8000/Rongerr.github.io/pdf_files/1_0_dilatedConv.pdf

远程仓库的路径：https://dearrongerr.github.io/Rongerr.github.io/pdf_files/1_0_dilatedConv.pdf

html 中 a 标签找的地址：设置是 `<a class="down-button" target="_blank" href="/pdf_files/1_0_dilatedConv.pdf" markdown="1">:fontawesome-solid-download: 下载</a>`，实际找到的是 ：127.0.0.1.8000/pdf_files/1_0_dilatedConv.pdf

html 中 a 标签设置的路径路径：`<a class="down-button" target="_blank" href="../pdf_files/1_0_dilatedConv.pdf" markdown="1">:fontawesome-solid-download: 下载</a>`，实际找到的是：127.0.0.1.8000/Rongerr.github.io/logs/pdf_files/1_0_dilatedConv.pdf

解决方法：

> 场景描述：
>
> pwd：`docs/logs/3_test.md`
>
> 要引用的文件路径：`docs/pdf_files/1_0_dilatedConv.pdf`

==（正确设置引用路径）== 使用`[]()` 找路径设置链接时，测试正确跳转 [点击跳转](../pdf_files/1_0_dilatedConv.pdf)  ，路径设置

>  `[点击跳转](../pdf_files/1_0_dilatedConv.pdf)`  

🟢 mkdocs serve 中解析的路径为：👇 ，并且可以正常打开

> `127.0.0.1.8000/Rongerr.github.io/pdf_files/1_0_dilatedConv.pdf`

因为在配置文件中设置的路径 `site_url` ：

```yaml
site_url: https://dearrongerr.github.io/Rongerr.github.io
```

🟢 部署到远程仓库，上传 gitpages，这路径被解析为：



==（a标签中正确设置引用路径）== 但是 a 标签中，如果想正确的引用，路径要被设置为 <a href=" /Rongerr.github.io/pdf_files/1_0_dilatedConv.pdf ">测试点击正确跳转</a>：

```html
href="/Rongerr.github.io/pdf_files/1_0_dilatedConv.pdf" 
```

