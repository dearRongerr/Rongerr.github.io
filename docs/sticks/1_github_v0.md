# git init

初始化本地仓库、后面推送远程

第一步：

```bash
git init
git add .
git commit -m "Initial commit"
# 可能需要
git config user.name "dearRongerr"
git config user.email "1939472345@qq.com"
```

第二步：重要）编写 `.gitignore`  

```bash

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
env/
ENV/
.env
.venv

# Jupyter Notebook
.ipynb_checkpoints

# 数据文件 (根据需要取消注释)
*.csv
*.xls
*.xlsx
*.parquet
*.feather
*.pickle
*.pkl

# 编译和缓存
__pycache__/
*.py[cod]
*$py.class

# 日志文件
*.log
logs/

# OS生成的文件
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE相关文件
.idea/
.vscode/
*.swp
*.swo
*~

# 模型文件和检查点 (根据需要取消注释)
# *.h5
# *.pkl
# *.pt
# *.pth
# checkpoints/
# models/

# 图像和大型媒体文件 (根据需要取消注释)
# *.png
# *.jpg
# *.jpeg
# *.gif
# *.mp4
# *.mov

# 环境配置文件
.env
.env.local
config.ini
```

尤其注意 ：

（1）如果已经提交了 `.DS_Store` ，请执行：

```bash
git rm --cached **/.DS_Store
```

（2）接着再次提交