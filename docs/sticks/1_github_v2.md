# git分支与功能测试

## function分支合并到 main 分支

> 场景描述：main 分支保存当前最好最新的状态
>
> function分支测试，测试成功合并到主分支

（1）从当前main位置创建function分支

```bash
git checkout -b function
```

（2）验证分支状态

```bash
# 确认main分支正确指向"添加小波变换"提交
git checkout main
git log -1 --oneline  # 应显示"添加小波变换"

# 确认function分支与main一致
git checkout function
git log -1 --oneline  # 也应显示"添加小波变换"
```

（3）后续在 function 分支上工作：

```bash
# 在function分支上进行所有开发工作
git checkout function

# 修改代码
# ...

# 提交更改
git add .
git commit -m "在function分支上的改进"

# 如果改进成功，可以将function合并到main
git checkout main
git merge function

# 如果需要继续在function上开发新功能
git checkout function
# 继续开发...
```

## 回退提交、修改头指针

> 场景描述：修改main 头指针，指向之前的某次提交

（1）查看想要回退提交的哈希值

```bash
git log --oneline
```

（2）将main分支指向该提交

```bash
git checkout main
# 通过哈希值回退
git reset --hard <提交哈希值>
# 通过引用回退：指向 当前HEAD的前一个提交
git reset --hard HEAD~1
```

> 场景：完整的工作流

## 保留功能分支的完整Git工作流

（1）创建function 分支

```bash
# 确认main分支位于"添加小波变换"提交
git checkout main

# 创建并切换到function分支
git checkout -b function
```

（2）在function分支上进行实验

```bash
# 确保在function分支上
git checkout function

# 修改代码（例如调整etth1_u.sh中的参数）
nano scripts/etth1_u.sh
# 修改dropout参数或添加新功能

# 提交更改
git add scripts/etth1_u.sh
git commit -m "实验: 调整dropout参数为0.3"
```

（3）如果实验结果良好，合并到main分支

```bash
# 切换到main分支
git checkout main

# 合并function分支（使用--no-ff保留合并历史）
git merge --no-ff function -m "合并improved-dropout实验: 性能提升10%"

# 可选: 为此版本添加标签
git tag -a v1.1 -m "ETTh1数据集上的改进版本: MSE降低到0.xxx"
```

（4）继续在function分支上进行新实验

```bash
# 切换回function分支
git checkout function

# 可选: 将function重置到与main一致（如果想基于最新main开始）
git reset --hard main  

# 进行新的修改.....
# 例如添加新的正则化层

# 提交新实验
git add models/Time_Unet.py
git commit -m "实验: 添加dropout到高频处理部分"

# 运行新实验
bash scripts/etth1_u.sh
```

（5） 有用的辅助命令

```bash
# 查看当前状态
git status

# 显示提交信息
git log

# 如果想放弃当前实验
git checkout function
git reset --hard main

# 如果想创建新的实验分支
git checkout main
git checkout -b new-experiment
```

 多分支实验

```bash
# 从main创建多个实验分支
git checkout main
git checkout -b exp-regularization
git checkout main
git checkout -b exp-architecture

# 在不同分支测试不同思路
# ...
```

