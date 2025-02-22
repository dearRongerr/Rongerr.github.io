# MkDocs

![image-20241115100605111](mkdocs_learn/image-20241115100605111-1636372-1636377.png)

主题配置：[**Material for MkDocs**](https://squidfunk.github.io/mkdocs-material/getting-started/)

本地调试：

```
(base) .. mkdocs-site % mkdocs -h
Usage: mkdocs [OPTIONS] COMMAND [ARGS]...

  MkDocs - Project documentation with Markdown.

Options:
  -V, --version         Show the version and exit.
  -q, --quiet           Silence warnings
  -v, --verbose         Enable verbose output
  --color / --no-color  Force enable or disable color and wrapping for the output. Default is auto-
                        detect.
  -h, --help            Show this message and exit.

Commands:
  build      Build the MkDocs documentation.
  get-deps   Show required PyPI packages inferred from plugins in mkdocs.yml.
  gh-deploy  Deploy your documentation to GitHub Pages.
  new        Create a new MkDocs project.
  serve      Run the builtin development server.
```

[参考模版源码](https://github.com/Yang-Xijie/yang-xijie.github.io)

[参考模版展示](https://yang-xijie.github.io/)

[官方文档：mkdocs配置 ](https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/)

[mkdocs入门教程]( https://b23.tv/jQs24a5)

## 文件组织形式

```bash
(base) ... docs % tree
.
├── Error  # 文件夹
│   └── 报错.md   # markdown文件
├── Leecode
│   └── 力扣.md
├── home
│   ├── page-1.md
│   └── page-2.md
├── index.md
├── mkdocs
│   ├── css
│   │   ├── no-footer.css
│   │   └── unordered-list-symbols.css
│   └── javascripts
│       └── katex.js
└── 便签  # 文件夹
  ├── TODO  # 图床
  │   ├── 1.png
  │   └── image-20241115095446260.png
  ├── TODO.md #markdown文件
  ├── mkdocs_learn
  │   └── image-20241115100605111-1636372-1636377.png
  ├── mkdocs_learn.md
  └── 备忘.md

10 directories, 14 files
```

前段与后端的对应

![image-20241115101310759](mkdocs_learn/image-20241115101310759.png)

## 添加页面创建时间、最后一次修改时间

[官方文档链接](https://squidfunk.github.io/mkdocs-material/setup/adding-a-git-repository/#code-actions)

![image-20241115101535524](mkdocs_learn/image-20241115101535524.png)


## 写作

更多写作

```
!!! note
    This is a note.
```

```
!!! tip
    This is a tip.
```

```
!!! warning
    This is a warning.
```

```
!!! danger
    This is a danger.
```

```
!!! success
    This is a success.
```

```
!!! info
    This is a info.
```

```
!!! quote
    This is a quote.
```

```
??? question "What is the meaning of life, the universe, and everything?"
```

!!! note
    This is a note.

!!! tip
    This is a tip.

!!! warning
    This is a warning.

!!! danger
    This is a danger.

!!! success
    This is a success.

!!! info
    This is a info.

!!! quote
    This is a quote.

??? question "What is the meaning of life, the universe, and everything?"

## mkdocs命令

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

**Project layout**

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

## 一些修改

- 本地文件和在线文件的存储问题，上传上去的本地怎么管理，又不能完全在线

等你写得多到占用本地太多空间再说吧，笑）

- 
   图床 & typora& vscode&github

typora 可以自动创建图床文件夹

- 
   [mkdocs material 内容差参考](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/#highlighting-specific-lines-lines)
- 
   [时间戳显示有问题 github actions error](https://zhuanlan.zhihu.com/p/688321385)
- 
   版本修改

[官方链接](https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/)

[版本控制示例](https://mkdocs-material.github.io/example-versioning/latest/)

[版本控制源码](https://github.com/mkdocs-material/example-versioning)

好复杂，再说吧

- 
   文档标题加编号（可以但没必要，新建CSS文件，然后在yml配置文件中引用
- 
   mkdocs的文件组织结构

docs/文件夹（导航栏）/（起个别名）/文件夹/文件夹/md文件

docs/文件夹（导航栏）/文件夹（左侧栏下拉条）/md文件

docs/文件夹（导航横栏）/md文件（左侧栏）/一级标题（标题处）/二级标题（目录从二级标题开始显示）

一级标题直接会显示在左侧栏，或者在yml文件中起别名

- 
   英文文本 两端对齐(以后再说吧，人家都没弄，我也不折腾了)
- 
   这个[主题](https://wcowin.work/)超好看，有空折腾一下
- 
   git push origin main每次push就会把所有文件的时间全部更改了

改对了！重新把整个 [工作流文件](https://wcowin.work/Mkdocs-Wcowin/blog/websitebeauty/time/)复制了别人的一份。

- 
   文件结构变了，记得修改yml的路径
