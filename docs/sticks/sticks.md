## git命令

新建仓库：

```python
echo "# Rongerr.github.io" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/dearRongerr/Rongerr.github.io.git
git push -u origin main
```

向已有仓库推送

```python
git remote add origin https://github.com/dearRongerr/Rongerr.github.io.git
git branch -M main
git push -u origin main
```

## mkdocs命令

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

**Project layout**

​    mkdocs.yml    # The configuration file.
​    docs/
​        index.md  # The documentation homepage.
​        ...       # Other markdown pages, images and other files.

## 终端命令

-  `cd ..`  返回上级文件
- `ls`  显示当前目录文件
- `ls -a` 显示当前目录的所有文件，包括隐藏文件

**macOS 终端命令**

- 显示文件树

  - 打开mac终端

    - 输入 brew install tree

    - 使用：

      `tree` 显示文件树

      `tree -a` 显示所有文件树，包含隐藏文件

- vim？





