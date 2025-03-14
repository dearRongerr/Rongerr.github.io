# 乐观 & 坚强

## 2025 年 3 月 14 日

真好啊，真好啊，SegRNN 的代码看完了，下面还有一些细节部分，模型的损失什么的，主体部分都看好了。下面一步，就是完全跑完论文，看看实验结果，包括消融实验的部分。顺着实验：

![image-20250314180636133](images/image-20250314180636133.png)

把这边的几个模型都跑通，总而言之，就是收尾工作，复现论文中的所有结果。

把这个项目完全看好了，后面也都好说，因为时间序列的代码确实很多一样的。

今天忽然意识到，没必要特别关注类在哪个文件中定义的，知道调用的哪个类和类实现的功能更重要。

时间序列中残差连接。

完全弄明白一个代码也挺难的，还有很多小点不明白。

## 2025年 3 月 13 日

我想我会一直坚强，就像我爷爷一样。我会一直一直想念这个倔强不服输的老头儿。爸爸妈妈教会的我的是慈悲，永远温和。那么我爷爷身体力行的是 倔强。okay，加油。

## 2025 年 3 月 7 日

(1) 改这个什么指纹的错误。原因：没有退出 debug 直接断网，就会报错 `git remote -v `，删了 github 本地仓库

（2）git push 忘了明明是啥了 `git push -u origin main`

更新：

谁的代码又报错啦？耶，是我

![image-20250307183921933](images/image-20250307183921933.png)

---

更新：

刚刚又因为断网，导致错误了。就是说还是正常的吧断开连接和退出吧，已经吃过两次亏了，也许是因为这次把重定向删除了，所以没有报进程的错误，只是报了端口被监听的错误

```
| tee
```



```
>
```



---

留存正确的 shell 调用 python 脚本：

```
model_name=SegRNN

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=illness
data_name=custom


seq_len=60
for pred_len in 24 36 48 60
do
    python -m debugpy --listen 5998 --wait-for-client run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len 12 \
      --enc_in 7 \
      --d_model 512 \
      --dropout 0 \
      --train_epochs 30 \
      --patience 10 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --revin 1 \
      --itr 1 --batch_size 16 --learning_rate 0.001 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
```

以及 launch.json的配置：

```
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "[这里更换为任意名称]",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5998
            }
        },
        {
            "type": "bashdb",
            "request": "launch",
            "name": "Bash-Debug (type in script name)",
            "cwd": "${workspaceFolder}",
            "program": "${command:AskForScriptName}",
            "args": []
        },
        {
            "type": "bashdb",
            "request": "launch",
            "name": "Bash-Debug (select script from list of sh files)",
            "cwd": "${workspaceFolder}",
            "program": "${command:SelectScriptName}",
            "args": []
        },
        {
            "type": "bashdb",
            "request": "launch",
            "name": "Bash-Debug (hardcoded script name)",
            "cwd": "${workspaceFolder}",
            "program": "${workspaceFolder}/path/to/script.sh",
            "args": []
        },
        {
            "type": "bashdb",
            "request": "launch",
            "name": "Bash-Debug (simplest configuration)",
            "program": "${file}"
        }
    ]
}
```

以及最后别忘了正常打断点

---

## 2025 年 3 月 6 日

人麻了，谢谢，调代码都懂得一张图：程序能跑就别动，可惜我打了那么多超级用心的记录点，不说了，重新调代码了。算了，也有收获的

看明白的地方都变成记录点，有收获的

感谢自己是个老冗余的人了，全部复制保存了一份，还有救。今天学习参数 `--wait-for-client` 和 `--itr 1 --batch_size 16 --learning_rate 0.001 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log`

 `| tee`参数

崩溃的时候，只恨自己没搞个版本库回退状态了。算了，不会，再说。

应该是中断再加上重定向的问题。

```
model_name=SegRNN

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=illness
data_name=custom


seq_len=60
for pred_len in 24 36 48 60
do
    python -m debugpy --listen 5998 --wait-for-client run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len 12 \
      --enc_in 7 \
      --d_model 512 \
      --dropout 0 \
      --train_epochs 30 \
      --patience 10 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --revin 1 \
      --itr 1 --batch_size 16 --learning_rate 0.001 | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
```

## 2025年 3 月 5 日

大概就是存一些活着的痕迹；我能每天说 100 句丧气话，然后第二天依旧活力满满。因为是自己的选择，为自己的选择绝对的负责。做不下去换就是了。Okay，Done。

----

## 2025 年 3 月 1 日

3 月，加油加油

新技能：vscode 中，shell 调用 python 文件，怎么调试，属实巧合，心平气和的分析，问题提出来就是能解决

yepyep

![image-20250301213316603](images/image-20250301213316603.png)

----

## 2025 年 2 月 28 日

寄了，跑不动，换换吧

复习复习，命令总也不用 该忘了

![image-20250228141514138](images/image-20250228141514138.png)

---

## 2025 年 2 月 27 日

关于 docker脑子里大概有一整套流程了

打算使用 docker 封装代码，增加可移植性，好心的同门愿意帮我跑，具体地做法：

- docker pull拉取一个官方的镜像
- docker ps 查看镜像
- docker tag 重命名，镜像名:标签
- docker 压缩成 tar 文件，命令忘了，docker save 
- 上传 tar 文件到服务器
- docker -i 路径 load 加载镜像
- docker ps查看
- 有了的话，docker run，启动一个容器，因为我主要想把自己的项目文件封装到容器中，再打包成tar文件也好，再分享到社区也好
- 大概需要目录挂载，把本地的文件挂载到容器内的某个文件，而不是卷映射，因为要以外部的文件为准
- 可能会遇到的错误，库不够，那就 pip 安装。

----

## 2025年 2 月 26 日

最近这几天把 git 的相关内容，最简单的系统的过了一遍，本地仓库和远程仓库是两个概念，知识的理解确实更加深刻了，git remote -v查看本地仓库连接的远程仓库的地址和别名，本地仓库的修改，git staus查看本地仓库的状态，然后一套组合拳，git add.，git commit -m"提交信息"，git push 推送到远程仓库，第一次在命令行执行，没有从 vscode 中傻瓜式操作了🫣也许这个世界不缺知道的人，缺掌握的人。okay，继续学习

![image-20250226165036090](images/image-20250226165036090.png)

![image-20250226165051423](images/image-20250226165051423.png)

![image-20250226165107964](images/image-20250226165107964.png) 

---

💕🌈🐾 但行好事，不问前程

🪐💫 捡垃圾的人不会嫉妒开宝马车的人，但会嫉妒垃圾比它捡得多的人

🌺🐋✨ 面朝大海，春暖花开

## 241219

<u>（241219）一整个大摆烂，你努力吧，我开心就好了。生命之树即将枯萎，灵魂马上窒息，我装不去下了</u>

- 241115 小红书上发了个贴，群起嘲之：别骂了别骂了，我错了
- 241117 来了
- 241118 来咯
  - [x] 文献阅读笔记
  - [x] 好消息：文章时间戳改对了
- 241119 开工，慢慢来也挺好的，是的
- 241125 干活
- 241126 来了
- 241127 来了
- 241128 星期四 来了 晴天
- 241129 星期五 来了 阴天

----

⭐️ Week 1

## 2024年12月1日 星期日 休

## **2024年12月2日 星期一 来了 晴天**

- [x] 查：[归一化](https://dearrongerr.github.io/Rongerr.github.io/learning/1/)
- [x] 改：归一化、文献阅读 COUNTGD、COUNTR
- [x] 增：GAN

8h9min

## **2024年12月3日 星期二 来了 晴天**

- [x] 改：GAN

6h34min

## **2024年12月4日 星期三 来了 晴天**

- [x] 改：GAN（DONE）
- [x] 改：ViT（DONE）
- [x] 增：Bert
- [x] 增：vision transformer代码（DONE）
- [x] 增：clip

10h45min

## **2024年12月5日 星期四 晴天 来了**

- [x] 改：WeightNorm

2h21min

2024年12月6日 星期五 教资面试

2024年12月7日 星期六 教资面试

## **2024年12月8日 星期日 阴天 来了**

- [x] 改：WeightNorm
- [x] 增：GAN的变体：最小二乘GAN

3h45min

----

⭐Week2

## **2024年12月9日 星期一 阴天 来了**

- [x] 改：GAN变体：cGAN（DONE）
- [x] 增：pytorch读取csv、excel文件转换成tensor
- [x] 增：ResNet项目实战
- [x] 改：Diffusion models

8h59min

## **2024年12月10日 星期二 下雪 来了**

- [x] 查：GAN变体
- [x] 改：DDPM
- [x] 增：KL散度（DONE）

6h44min

## **2024年12月11日 星期三 阴天 来了 冲！**

- [x] 改：DDPM推导

## **2024年12月12日 星期四 阴天 来了** 

- [x] 增：latex捉虫

## **2024年12月13日 星期五 晴天 来了**

- [x] 改：latex捉虫

----

⭐Week3

## **2024年12月19日 星期四 阴天 来了**

- [x] 改：扩散模型推导

## **2024年12月20日 星期五 早上好**

- [x] 改：VAE推导
- [x] 增：RNN

## **2024 年 12 月 21 日 星期六 上午好**

- [x] 增：小讲堂（ViT、CLIP）、YOLO、DETR、DINO、对比学习、GPT
- [x] 查：RNN（DONE）

> 近期默念：**「永远摆脱受害者叙事」**
>
> 停止向世界描述我的监狱
>
> 更重要的是：停止向自己描述
>
> 更重要的是：现在从监狱里站起身，走出去
>
> 因为监狱没有上锁，也没有门
>
> 最后，根本没有监狱
>
> 永远不要试图将自己打造成一个受害者来获得别人的理解和爱，不要觉得世界对你不公，不要觉得谁对不起你
>
> 把自己当成受害者，就永远要找凶手；
>
> 把自己当成前进者，就永远在找助手

## **2024 年 12 月 22 日 星期日  上午好 晴天**

- [x] 查：LSTM
- [x] 改：MOCO

---

⭐Week4

## **2024 年 12 月 23 日 星期一  中午好 阴天** 

- [x] 查：LSTM

## **2025 年 1 月 21 日 星期二 22:16**

此时此刻我终于是回到了家，若是掐指一算，也是过去一个月了。

## **2025 年 2 月 20 日 星期四 21.00**

今天心情不错，我这个电脑的快捷键设计的太舒服了

option+1 typora

option2 edge 浏览器

option3 vscode

option5 网易云

option7 safari

command7 google

option9 wps

^1 桌面

桌面 1 ssh vscode

桌面 2 vscode 代码学习

option `截图，optionT 贴图

...很顺滑

今天在跑代码

![image-20250220211529296](images/image-20250220211529296.png)

option0 设置

command0 访达