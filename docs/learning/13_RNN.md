# RNN

ref：[【29、PyTorch RNN的原理及其手写复现】](https://www.bilibili.com/video/BV13i4y1R7jB/?share_source=copy_web&vd_source=5cbbeafd6fa2338b041c25f100ea6483)

![image-20241220115043569](images/image-20241220115043569.png)

topic：

（1）不同类型的RNN的图示以及应用场景的图示

（2）介绍pytorch中RNN的api的使用

（3）通过代码验证 RNN 内部是如何计算的，通过代码来 验证 pytorch的RNN的api 并对比结果

## k1 记忆单元分类

![image-20241220115210956](images/image-20241220115210956.png)

- [x] 什么是记忆单元？

记忆单元就是 存储的 过去的历史信息

- [x] 什么是循环神经网络？

所谓循环神经网络 就是说，在对序列进行建模的时候，在算每一时刻的表征的时候，一般考虑过去的 历史信息。这个历史信息 就是通过 记忆单元 保存的。然后每个时刻 我们都会从 记忆单元中 获取 过去的 历史信息，然后辅助当前时刻 做预测。

- [x] 记忆单元分类

关于记忆单元 一般有三类

1. RNN
2. LSTM
3. GRU 

> 一类 比如说 RNN，比如说 Simple RNN，简单的RNN 结构，等下实现的也是 简单的RNN结构
>
> 另外两种是 GRU和LSTM，这两种网络的记忆性会更强一点；计算复杂度也会更高一点；使用频率也会更高一点，就是说 现在很多的实际应用中，我们基本使用的是LSTM或者GRU；但是它们都是RNN的一个变体，所以RNN是基础；

## 	k2 模型的分类

![image-20241220115515504](images/image-20241220115515504.png)

（1）单向循环

> 循环神经网络也可分为单向循环，所谓单向循环就是，当前时刻的预测 只跟 过去有关，从左到右 递归的计算。

（2）双向循环

> 双向循环，双向循环就是说 不只有 从左到右的 也有 从右到左的，就是说有两条链，另外一条链，在计算当前时刻的预测的时候 会考虑 未来信息。

（3）多个单向 、 多个双向

> 这个就是双向循环；那还可以把 多个单向 或者说 多个双向 叠加起来，也就是deep RNN 深度循环神经网络

![image-20241220115718028](images/image-20241220115718028.png)

（1）单向的循环神经网络

![image-20241220115802397](images/image-20241220115802397.png)

可以分为三层：

1. 最下面一层是 input layer，也就是输入层；
2. 中间是隐含层；
3. 最后是输出层；

> 下面的输入层每一个神经元 可以看做 每一个时刻；
>
> 也就是说 每一个时刻 不仅跟当前时刻的输入有关，还跟上一时刻的记忆单元有关；
>
> 并且在单向循环神经网络 中 始终是 从左到右的；
>
> 就是说当前时刻的预测 只跟 过去的记忆单元 有关，跟未来的 是无关的；

（2）双向的循环神经网络

![image-20241220120003057](images/image-20241220120003057.png)

1. 有两条链
2. 分为4个部分：  **input layer、output layer、forward layer、backward layer**
3. **（forward layer）**  forward layer是从左到右的循环 ，意思就是说 在 forward layer的输出中，它的输出不仅跟当前输入有关 也跟过去的记忆单元有关；
4. **（backward layer）**  backward layer当中，它的当前时刻的输出 不仅跟当前时刻的输入有关，还跟未来时刻的记忆单元有关，所以是 从右到左的 递归运算的。
5. **（将forward和backward结合）**起来有什么好处呢？ 就是说 既能看到过去 又能看到未来

## 	k3 语音识别模型性能比较

![image-20241220120329113](images/image-20241220120329113.png)

> 这张表格 来自某篇论文，这张表格 很好的 展示了 RNN、LSTM、 双向 单向、MLP、以及是否delay等 在参数数量相等的情况下 在语音识别上的表现；可以看到 第二列 第三列 分别是训练误差和测试误差；
>

 通过表格 可以看到 不同的模型在 语音识别 这种 序列建模，序列分类这个任务上的表现

![image-20241220123407253](images/image-20241220123407253.png)

（1）第一行是MLP，MLP就是简单的DNN 是no window的（什么意思？）

> 我们把语音 分成很多帧，比方说一帧是 15毫秒 或者 20毫秒，对于每一帧 提取一个特征 比如说 傅里叶变换 得到一个频谱特征，然后 我们对每一帧 进行单独建模，所谓 no window就是 我们不考虑 周围的帧，只考虑 当前这个15毫秒，然后 我们 把它送入 DNN中，来去 进行一个 预测 分类，这样做的话 它的 训练误差 和测试误差 大概都是在 40% 左右；

![image-20241220120448060](images/image-20241220120448060.png)

（2）（10 frame window、stride）

> 第二行 MLP 10帧作为一个窗 意思是 我们现在 同样还是MLP，但是 现在MLP的 输入 不仅是 只有一帧的特征，而是把 每10帧 放到一起，那么这里是否有stride，就是说 这10帧 到底有没有交叠 并没有介绍，总之 第二行这个 输入 比 第一行 覆盖的 时间窗口 会更大一点 ；
>
> 那么这样可以看到 这个误差，显著的从 46% 降到 32%，这个结果说明 在语音识别 这个序列建模 任务中，当我们把 上下文特征 一起考虑的话 效果会 更好；这是第二行。

![image-20241220123237089](images/image-20241220123237089.png)

**（3）delay**

第三行，将MLP换成了 循环神经网络，一个简单的RNN 模型，并且括号 delay 0，等下会解释 什么叫delay，这里的意思就说，就是说 把 每一帧特征 像 第一幅图一样，比如说

![image-20241220123710787](images/image-20241220123710787.png)

这里是第一帧的特征，这里是第二帧的特征，这里是第三帧的特征，我们把每一帧的特征 送入到RNN中，通过中间的隐含层 对历史信息 进行更新，这样的网络 错误率也是相比MLP 更进一步，看到训练误差到30%，测试误差是35%，相比于上面 10帧的MLP，效果更好。

**（4）LSTM**

![image-20241220123805856](images/image-20241220123805856.png)

- 接下来 如果我们把RNN，替换成LSTM，效果更进一步
- 都是delay 0


**（5）LSTM+backwards**

![image-20241220123952932](images/image-20241220123952932.png)

再下面一步，还是LSTM，只是把输入 翻转过来，也就是把input序列倒过来，再输入到网络中，误差是差不多的，所以 仅仅是一条链的话，不论是正向识别，还是反向识别 其实效果是差不多的

**（6）RNN delay 3**

![image-20241220124429146](images/image-20241220124429146.png)

对输入进行改造，首先可以看到 同样是用 RNN网络，这里 对它 进行 delay 三帧，然后可以看到 它的效果 相比于原本的 RNN 从30% 降低到 29%，测试误差 也是从 35% 降低到 34%；

- [x] 那么这个 delay 3 帧是什么意思呢？

![image-20241220124532087](images/image-20241220124532087.png)

delay 3 帧的意思就是说，当 喂入 三帧 作为 输入的时候，前面 这三个输出，先不要，

就是说 先拿 三帧输入 送入到网络中 让它先对记忆单元 去 更新三步 ，然后到第四步（帧）的 输入的时候，才 把 输出拿出来， 作为 第一帧的预测值，这个就是delay 3的意思

- [x] 为什么 delay 3 帧效果有效？

如果 不做delay的话 ，在 输入 第一帧的 特征的时候，它的预测的输出 只能 看到当前的第一帧，范围就很小；

但是当 delay 三帧的时候 预测第一帧的输出 其实就看到了 三帧，它看到了 第一帧、第二帧、第三帧 都进入了 记忆单元中；

以上就是 delay RNN的结构；

- [x] 再次解释 delay

 delay 能够在 短暂 的 牺牲 时延的情况下，提高精度，看到更宽的上下文

> 有 delay 的话，在预测第一帧的输出的时候 肯定会 稍微 延迟一点，因为 如果 不做 delay的话，我们就直接 算一步就好了，如果delay 三帧的话，那在预测第一帧的输出的时候，需要计算 三步，所以肯定会有 一定时延的。但是这个时延 确实能够 使得 预测的效果更好，因为它看到的上下文 会 更宽一点；以上是delay的意思。

**（7）B** 

![image-20241220124913764](images/image-20241220124913764.png)

**双向的LSTM、RNN**

![image-20241220124945634](images/image-20241220124945634.png)

-  RNN delay三帧 和LSTM delay 五帧 效果都有不同程度的增加；
- 双向的结果比delay 和 单向的 效果都要好；
- 训练集 错误率从29%降低到24%；
- 测试集错误率也是明显降低；

<u>双向、delay</u> 

- 表示 看到了未来的信息；
- 当 delay三帧的话，在预测第一帧的输出的时候 其实是看到了第二帧、第三帧、第四帧  指的是 看到了未来的三帧的
- 当预测 第二帧的输出的时候 同样 看到第三帧、第四帧、第五帧
- 虽然也看到了未来的信息，但看到未来的信息还是不够长；

- 如果把单向 换成双向的网络的话，那么整个未来的特征 和 过去的 特征，网络都能看到，这就是说双向的范围 更大一点；

> - 单向delay 3：输出第一帧看到的是 输入第一帧、第二帧、第三帧
>
> - 双向delaye 3：输出第一帧，看到的是第一帧、第二帧、第三帧+第四帧、第五帧、第六帧 

<u>双向的缺点</u>

需要完全的 把整个输入特征序列 送入到网络中 ，最后才能得到输出

> 而单向带时延的情况就不需要把整个特征 都算出来 才能预测第一帧，只要有三帧了，就可以预测第一帧了；
>
> 所以单向带时延的，响应速度会更快；
>
> 双向的响应速度肯定是最慢的；
>
> 所以在速度 和效果上 需要 取得一个比较好的平衡 才能满足具体的业务需求。

## 	k4 循环神经网络的优缺点

**一、优点**

> （1）权重共享可以处理变长序列
>
> （2）模型的大小 与 序列长度无关
>
> （3）计算量与序列长度呈现线性关系
>
> （4）考虑历史信息
>
> （5）便于流式输出
>
> （6）权重时不变

**二、缺点**

> （1）串行计算速度慢
>
> （2）无法获取太长的历史信息

**第一点**

优点可以处理变长序列

> 这个是DNN和CNN处理不了的，比如DNN，输入的特征是固定的，而CNN的不仅和kernel size有关，还跟输入的通道数有关，所以如果CNN 输入通道数有变化的话 还需要重新搭建一个网络，而RNN 是可以处理变长序列的

- [x] 为什么RNN 能处理变长序列呢？

![image-20241220125908343](images/image-20241220125908343.png)

原因是因为，可以看到图中 有一个w

> 也就是 权重，这个w在每个时刻 都是相等的，正是因为 所有的权重，在每一个时刻都是相等的；不论是 输入 跟既有单元的连接，还是历史信息 跟当前的神经元的连接 它的权重都是固定的，正是因为 权重 在每一时刻 共享，所以 RNN 能够处理变长序列；

一旦去掉了 <u>权重 共享</u> 这个归纳偏置的话，就是说，如果每一时刻 都有一个 不一样的 w的话，这个时候 就不能处理 变长序列了，就类似 position embedding 一样，只要遇到了 长度 比训练集大的，那就处理不了了（也不是，三角变换）；

**第二点**

![image-20241220130243673](images/image-20241220130243673.png)

第二点，模型的大小 与 序列长度无关，这里说的是 模型的大小，是说模型的参数数量 与 长度无关，模型的全部参数 和序列长度 都是无关的，只输入特征 和输入通道数 以及RNN的隐含单元有关

**第三点**

![image-20241220130406832](images/image-20241220130406832.png)

第三个优点就是 RNN的计算量 跟 序列长度 呈线性增长，类比Transformer，在原本的Transformer中 最大的一个 诟病的地方 就是 计算复杂度 跟序列长度 是呈一个平方关系的，但是在RNN中，计算量 是跟长度 呈现 线性增长的；

> 举例子：
>
> 当 序列长度 为2的 时候，计算量 可能就是2t
>
> （t指的是时间？模型 固有的计算量）
>
> 当序列长度为3 的时候，计算量 就是3t，就不是说 从 4变成9，呈现 平方关系。
>
> 在RNN中 呈现 线性关系；这是跟 Transformer 在计算量上 一个明显的区别。

**第四点**

![image-20241220130603036](images/image-20241220130603036.png)

相比DNN而言，RNN是可以考虑到 历史信息的，因为有链式的结构，可以通过隐含层 来积累 历史信息；

**第五点**

![image-20241220130731325](images/image-20241220130731325.png)

流式 输出，可以看到：

![image-20241220130819372](images/image-20241220130819372.png)

- [x] 流式输出是什么？

每 计算一步，都可以得到 一个输出，这个输出 可以直接 送给 用户，这就是 流式 的意思。

> 但是对于 Transformer而言的话，由于它是考虑到全局的信息 计算一个 全局的self attention，所以就不能单步 的计算 每一步的 输出，这就是 Transformer的一个缺点，不能直接的 应用到 流式的场景；
>
> 但是在循环神经网络中，只要每算一次 递归运算，就可以得到一个输出，这个 输出就可以直接返回给用户，这就是流式的，也就是 不需要 把 整个序列 都算完 才返回给用户，而是说 每算出一个 时刻 都可以返回给用户

**第六点**

![image-20241220131059489](images/image-20241220131059489.png)

权重时不变

> 权重是 时不变的，正是因为RNN 权重 时不变，所以RNN 可以处理 变长序列；

**二、缺点**

![image-20241220131203763](images/image-20241220131203763.png)

- [x] 为什么说 串行计算慢

因为 在算 每一时刻的时候 都需要等 上一时刻的历史信息，等上一时刻的算出来 才能算 下一时刻，是一个 串行的过程，比较慢

- [x] 怎么理解 RNN 也是无法获取太长的历史信息

也就是说 由于梯度消失的问题，导致RNN无法 从当前时刻 获取很久远的信息

> RNN 由于梯度消失的问题，无法获得太长的历史信息。
>
> 这一点正是Transformer的优点。
>
> Transformer的归纳偏置 是比较弱的，是通过一个 全局的self attention，来计算 两两位置之间的一个相关性，所以Transformer是可以上下去捕捉 很长的历史关联性的。

## 	k5 RNN 的应用场景

![image-20241220131740414](images/image-20241220131740414.png)

**（1）生成任务**

生成任务，比如歌词生成、对联生成、像GPT一样写小说

生成任务，如果用一幅图来表示：

![image-20241220131838603](images/image-20241220131838603.png)

1、如图表示RNN在诗歌、语音、符号生成中的表示

2、这类任务可以看成one to many的过程，也就是说 只要给了 一个输入，或者一个很短的 输入，RNN就可以利用自己的 递归机制 不断的预测 新的输出，就比如 给出 一两句话，RNN 写出一段话 或者 一篇文章，就是 one to many，RNN在生成任务上的应用

**（2）情感分类**

RNN也能做情感分类

> 比如说很古老的一个情感分类任务，对影评进行分类，判断一句话是正向情感还是负向情感，对于一个情感分类任务，可以看成many to one的任务

![image-20241220134939061](images/image-20241220134939061.png)

输入是一段话或者说一篇文章，但是输出 只有一个，只需要对一段话预测一个类别就好了，这个就是many to one的任务，典型的应用场景就是去情感分类

![image-20241220135039924](images/image-20241220135039924.png)

many to many的任务：

- 词法识别
- 机器翻译

词法识别就是识别当前这个词是名词还是动词，当前这个单词多音字等等

机器翻译，在Transformer中是应用比较多的；

但是这两种 many  to many的模型结构还是有一些区别的，可以看到下面两幅图：

（一）词法识别

![image-20241220135218541](images/image-20241220135218541.png)

- 识别一句话中，每个字的拼音是什么，或者识别每个词的词性，这种就是many to many

- 属于直进直出的many to many

（二）机器翻译

![image-20241220135306632](images/image-20241220135306632.png)

- sequence to sequence 结构；
- 有编码器，有解码器，中间依靠注意力机制，来帮助解码器预测每一时刻的输出，也是many to many；
- 常见的应用场景：机器翻译、语音合成等

![image-20241220135405098](images/image-20241220135405098.png)

语言模型 RNNLM；

总之就是

- one to one
- Many to one
- many to many

## k6  RNN框图

![image-20241220135618631](images/image-20241220135618631.png)

## torch.nn.RNN

![image-20241220135729914](images/image-20241220135729914.png)

- 可以用来构造一层 或者多层 简单的RNN结构； 

- RNN还有另外一种结构：激活函数，可以用tanh激活函数 或者 ReLU激活函数，使得RNN有更强的非线性建模能力；

- [x] RNN 计算公式是什么呢？

![image-20241220135820029](images/image-20241220135820029.png)

- 每一时刻的输出，或者说每一时刻的状态

- 在简单RNN中，输出是等于状态的， $h_t$也就是 $t$ 时刻的输出；

- 或者说 t 时刻RNN的状态 等于 tanh函数，就是非线性激活函数，里面分别是 $W_{ih}×x_t$ 再加上 $b_{ih}$，那么这里的$x_t$，就是当前时刻的输入，然后$w_{ih}$，就是在这个RNN中，它对输入的权重矩阵，就是 会用这个矩阵 来对权重 做一个映射，然后整体上，这个东西 可以看做 linear层，有权重 还有 偏置，$b_{ih}$，就是关于 权重的一个偏置

- 后面 还有一项，跟 历史状态有关的，跟 $h_{t-1}$ 有关的
- 也就是说，需要将 上一时刻的 输出 或者说 上一时刻的隐含状态 拿过来，然后对它进行一个 映射，用 $w_{hh}$ 的权重 来进行相乘，来进行映射，然后再加上一个偏置
- 总体而言 就是说 每一时刻的输出 或者说 隐含状态 不光跟当前时刻 的 输入 $x_t$ 有关，同时也跟上一时刻的记忆单元  $h_{t-1}$有关，并且都是线性组合的关系，最后通过一个非线性激活函数就能得到当前时刻的隐含状态；

![image-20241220140258663](images/image-20241220140258663.png)

- [x] 解释：

$h_t$ 是 $t$时刻的隐含状态

$x_t$是 t 时刻的输入

 $h_{t-1}$是  $t-1$时刻的隐含状态

$h_0$ 表示初始时刻的隐含状态

pytorch中也提供了两种 非线性激活函数：tanh和relu激活函数，默认用tanh激活函数

![image-20241220140410079](images/image-20241220140410079.png)

- 这是一个 class
- 在用RNN时候，首先要 实例化 这个class
- 实例化 class以后，得到RNN的一个模型
- 然后 再把 输入 喂入到 模型中，而不直接把 输入 喂入到模型中；
- 一般所有模型的 class，都需要 先进行一个实例化，然后才能得到一个layer；

- [x] 实例化RNN所需要的参数

![image-20241220140648773](images/image-20241220140648773.png)

- 第一个参数是 `input_size`,也就是 输入特征的大小，也就是 `x` 的特征的维度

- 第二个参数是 `hidden_size`，`hidden_size`决定了 $h_t$的大小，就是每一时刻的 $h_t$就是一个向量，对于单一样本而言，每一时刻 $h_t$就是一个向量，那么这个向量长度是多少呢？就是由 `hidden_size` 来决定

- 第三个参数 就是 `num_layers`，就是说 这个RNN，可以默认实例化的时候 只有一层，但是也可以改变 `num_layers`的值，变成多层，堆叠起来的结构，之前在介绍的时候也讲过，可以堆叠起来，单向的可以堆叠，双向的 也可 堆叠

- 第四个参数 就是 非线性激活函数，这里默认是`tanh`函数，也可以改成 `ReLu`函数
- 第五个是`bias` 一般会加上 这两个bias
- 第六个参数是 `batch first`，这个需要注意一下，这个参数就决定了 输入和输出的格式

> - 如果设置 `batch first=true`的话：
>
> 提供的输入张量 和 输出张量的 格式就是 `batch × sequence length×feature` 这样的格式
>
> 默认是`false`的，如果是 `false`的情况下：
>
> 需要保证 输入的格式是 `sequence length`，也就是序列长度 在第一个维度，`batch size`在第二个维度，`feature size`在第三个维度

- 第七个参数 `dropout`
- 最后一个参数`bidirectional`，最后一个参数 表示 双向的意思

> 也就是把这个参数设置为 `true`的话，就可以构建一个双向的RNN结构；
>
> 既然是 双向RNN结构，输出的特征大小就是`2×feature size`，就是2倍的`feature size`；

<u>双向结构图</u>

![image-20241220141145243](images/image-20241220141145243.png)

- 这幅图 就是 双向的，一旦把RNN设置成 双向的话，最终的输出 是由`forward输出`和`backward输出`一起拼起来的，所以这个 输出状态是 二倍的 `hidden size`，可以指定 `concat`和`sum`，一般用 `concat` 更多一点

- 也就是说 如果 设置 `hidden size是16`的话，那么 `output layer`大小，就是32，如果是双向的话

以上是RNN实例化的参数讲解；

- 当实例化完以后，就得到了RNN层
- 然后就可以 提供 输入 和 初始的隐含状态，来去递归的算出 每一时刻的 输入 所对应的输出是什么；

当实例化完 一个RNN，就可以 提供 `input` 和 $h_0$，来给出真正的输入序列：

![image-20241220141454221](images/image-20241220141454221.png)

- [x] 解释input

输入一般是三维的：

![image-20241220141550442](images/image-20241220141550442.png)

如果设置的`batch size first等于true`的话，那对应的输入格式就是 `batch size×sequence length×hidden size`；

反之 如果`没有设置batch size等于true`的话，提供的格式就是 `sequence length×batch size×hidden size`



- [x] 解释 $h_0$

- $h_0$的格式是 ($d×{num\_layers}$， $N$，$H_{out}$ )
-  $h_0$ 是 初始状态，只有 这一个时刻，所以这里不需要考虑 `sequence_length` 这个维度

- [x] 那这里也是 三个维度，为什么呢？

因为  RNN 可以是 多层 也可以是 双向，所以第一个维度 其实就是 是否是 双向  跟 多层 这两个因素 决定的；

`case1：`如果模型是一层，并且是单向的话，那么第一个维度 就是 1 ；

`case2：`如果是 有两层，并且是 单向的话，那么就是 1×2；

`case3：`如果是双向 并且是 两层的话，那就是 2×2=4；

所以这里的 第一个维度 $d \times num\_layers$ 由是否双向 以及 层数有关

![image-20241220142609695](images/image-20241220142609695.png)

第二个维度 $N$，就是 `batch size`，每个样本 都可以 设置一个 初始状态

第三个维度 $H_{out}$ 就是 `hidden size`的大小，因为 初始状态 就是一个向量,第三维 就是 向量的长度

## 	代码示例

### 	1 单层单向 RNN 

> 这个RNN 是一个 class
>
> 所以，首先实例化一个单向单层的RNN

step1：import  torch.nn as nn

step2：实例化 nn.RNN

step3：传入 实例化参数；

> - input_size=4
>
> - hidden_size也可以 随便写一个 hidden_size=3 
>
> - num_layers可以传入1
> - batch first设置成true
> - 定义为`single_rnn`

```python
import torch
import torch.nn as nn
# 1.单向、单层RNN
single_rnn = nn.RNN(input_size=4,hidden_size=3,num_layers=1,batch_first=True)
```

以上是 单层单向RNN，接下来构建一个输入

输入的维度一般是 `batch_size×sequence length×输入特征`

输入特征就是RNN实例化时的 `input size=4，batch size=1，sequence length=2，特征维度=4`

以上构建好了input序列，分别是： `batch_size × sequence length×输入特征`

```python
input = torch.randn(1,2,4) 
# batch_size*sequence_length*feature_size
```

把这个`input`作为 `single_rnn`的输入；

也可以不传入$h_0$,它默认以$0$向量填充

![image-20241220143813357](images/image-20241220143813357.png)

同时也可以看看 官网 api 输出是什么

![image-20241220143851387](images/image-20241220143851387.png)

输出是两个值，一个是整个的，所有时刻的输出；

另外一个输出的量就是最后一个时刻的状态，要定义变量接收输出

```python
output,h_n = single_rnn(input)
```

这样整个输出就算完了，接下来看一下$output$和 $h_n$

![image-20241220144015618](images/image-20241220144015618.png)

代码解读：

（1） `input`的形状 `1×2×4 = batch size×sequence length×feature dim`

（2）`single_rnn` 的参数含义：`4,3,1=input_size,hidden_size;num_layers`

（3）`output`大小就是 `1×2×3`

- 1表示 batch size，输入batch size=1，输出 batch size也是1，没有改变
- 2是 sequence length，序列长度，我们喂入的输入长度是2，所以输出的长度也是2
- 3，第三个维度为什么是3呢？因为我们设置的hidden size=3，也就是说 每个输出的状态向量 长度是3

（4）$h_n$： 最后一个时刻的隐含状态，在简单RNN中，最后一个时刻的隐含状态等于最后时刻的输出的，output最后一行的值 等于 $h_n$

![image-20241221102700631](images/image-20241221102700631.png)

### 	2 双向、单层RNN

```python
single_rnn = nn.RNN(input_size=4,hidden_size=3,num_layers=1,batch_first=True)
```

- input size不变

- hidden size不变
- num_layers不变
- batch first也不变
- 但是需要新增一个参数，叫做：

![image-20241221103009943](images/image-20241221103009943.png)

：bidirectional，这个参数默认是false，把它置成true

然后命名为 bidirectional_rnn：

```python
bidirectional_rnn = nn.RNN(input_size=4,hidden_size=3,num_layers=1,batch_first=True,bidirectional=True)
```

以上是实例化的双向RNN

- 输入特征大小是4
- 输出 or 隐含层大小是3
- 只有一层
- batch first=true
- 并且还是双向的

同样把上面的输入 送入双向RNN中，以`input`作为输入`bidirectional_rnn(input)`，因为无论双向、单向，输出都是一样的，都是`output` 和 `h_n`，表示区别加前缀`bi`

```python
bi_output,bi_h_n = bidirectional_rnn(input)
```

首先 打印 output的形状

```python
bi_output.shape
```

![image-20241221103251339](images/image-20241221103251339.png)

还有h_n的形状：

```python
bi_h_n.shape
```

![image-20241221103335046](images/image-20241221103335046.png)

对比，把单向单层RNN的output的形状，h_n的形状，都打印出来：

![image-20241221103356696](images/image-20241221103356696.png)

- 首先从输出上来讲：

（1）单向的输出大小是 1×2×3的

（2）双向的话变成了 1×2×6（一个batch size；2个sequence length；6个特征维度）

> 这是为什么呢？
>
> 这是因为在双向RNN中最后是把`forward layer`和`backward layer`两个输出拼起来，所以特征大小变成了两倍的`hidden size`；

- 最后一个时刻的状态也是不一样的

（1）在双向RNN中，它的维度是 2×1×3（前向的输出是个 1×3，后向的输出也是一个1×3）

（2）在单向中，维度是1×1×3

> 为什么呢？
>
> 因为双向中，其实是有两个层的最后一个时刻状态，有一个`forward layer`和一个`backward layer，`这两个状态在第一个维度上拼起来了，但是在单向中，只有一层的最后一个状态；

### 	3 RNN api 代码汇总

![image-20241221103816446](images/image-20241221103816446.png)

![image-20241221103830961](images/image-20241221103830961.png)

### 4 单向RNN&双向RNN 从矩阵运算的角度实现

注意：以下演示中，没有设置多层， num layers都定义的1层

（1）引入库，可以使用常见的pytorch函数

```python
import torch
import torch.nn as
```

（2）定义常量

然后定义一些常量，比如batch size、序列长度

```python
bs,T = 2,3  #批大小 和 序列长度
```

还需要定义 input size和hidden size，分别表示输入特征大小 和 隐含层 特征大小

```python
input_size,hidden_size=2,3 #输入特征大小，隐含层特征大小
```

有一个问题：怎么理解 时序模型中的 batchsize？

（3）生成 input

有了这些量以后，生成一个  input ，还是考虑batch first等于true的情况：第一个位置写batch size、第二个位置写序列长度、第三个位置写feature dim，也就是 input size

```python
input = torch.randn(bs,T,input_size) # 随机初始化一个输入特征序列
```

（4）初始化隐状态

初始化一个初始的隐含状态 `h_0`，初始的隐含状态一般是一个向量，如果考虑了`batch size`，就应该是 `batch size`个这样的状态，也可以先写成0：

```python
h_prev=torch.zeros(bs,hidden_size)  # 每一个状态向量大小是 hidden size
```

也就是在第一个时刻的时候，需要一个初始的隐含状态来，来作为第0时刻的初始状态

![image-20241221134624116](images/image-20241221134624116.png)

（5）调用pytorch RNN的API

还是用`nn.RNN()`的api，需要传入`input_size`，`hidden size`还有`batch first=True`，这样我们得到一个rnn

```python
rnn = nn.RNN(input_size,hidden_size,batch_first=True)
```

（6）传入参数

需要把 input 以及初始状态也传入RNN中，但是需要注意的是，api中初始状态是三维的

![image-20241221134803507](images/image-20241221134803507.png)

刚刚初始化的是 后面两维，第三维 我们没有初始化，因为这里是单向的 并且 只有一层的，所以对它扩一维就好了，扩0维，得到rnn output和h_finall，最后一个时刻的状态，或者叫state_finall

```python
rnn_output,state_finall = rnn(input,h_prev.unsqueeze(0))
```

这个是调用pytorch 官方的api，运行打印，看结果

![image-20241221134949743](images/image-20241221134949743.png)

（7）手写RNN forward 函数

定义`RNN forward`函数，实现RNN计算原理 `def rnn_forward():`，对于这个函数 首先要传入参数：

![image-20241221135123756](images/image-20241221135123756.png)

根据公式，要想算出$h_t$的话：

- 需要有$x$，$x$就是输入，所以第一个参数，需要写的是$input$
- 输入需要一个投影矩阵，就是$W_{ih}$，需要一个weight
- 同时还需要偏置项$\mathrm{bias_{ih}}$
- 还有上一时刻的隐含状态 ： $W_{hh}$ 
- 还有 $b_{hh}$
- 公式中还有 $h_{t-1}$ ，写成 `h_prev` ，就是前一时刻的状态

以上，就能算出RNN的输出

**第一步：获取当前时刻的输入特征得到`x`**

```python
def rnn_forward(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev):
```

- input 默认 三维的结构，先把input的形状拆解出来，形状应该是 `batch size×sequence length×input size` ，调用 `input.shape`

```python
bs,T,input_size = input.shape
```

- 通过拆解 `input` ，还可以知道 `hidden size`，也就是`h_dim`，也就是 `weight_ih`，可以根据它的权重所得到，也就是`weight_ih.shape`，那到底是`shape[0]`还是 `shape[1]`呢？看公式：

![image-20241221135636390](images/image-20241221135636390.png)

`weight ih`跟 `xt` 是左乘的关系，所以`weight`的第2个维度跟`x`是相同的，所以第一个维度 就是隐含单元的维度，所以写成`.shape[0]`，得到`hidden dim ：`

```python
h_dim = weight_ih.shape[0]
```

以上是得到了一些维度，接下来，可以写出 `h out`，首先 初始化一个 输出，输出大小是 `batch size×T×h dim`，初始化一个输出矩阵 或者 状态矩阵

```python
h_out = torch.zeros(bs,T,h_dim)  # 初始化一个输出（状态）矩阵
```

- `bs`跟输入是一样的

- 序列长度 或者叫 时间长度 也是跟 输入一样的维度
-  需要改成 `hidden size`这个维度

接下来 根据这 6 个参数，算出 `h out`

![image-20241221140038308](images/image-20241221140038308.png)

RNN是一个递归的计算，所以需要根据`x1`计算`h1`，根据`x2`计算`h2`等等，因此需要一个`for`循环 `for t in range(T):` 

因为RNN的计算复杂度 跟序列长度 呈线性关系，所以对序列长度进行遍历就好了

```python
for t in range(T):
```

首先得到当前时刻的输入向量，`input`，因为input是三维：

- 第一个维度是 batch size，全都取出来
- 第二个维度是时间，就拿当前 t 时刻的输入向量
- 第三维是特征维度，也是全部拿出来

```python
x = input[:,t,:]  # 获取当前时刻输入特征，bs*input_size
```

以上是第一步：获取当前时刻的输入特征得到`x`

第二步：扩充 batch 维度

根据公式，让`w`跟`x`进行相乘

- 这里`weight`一般默认传入 是二维的
- 而`x`的大小，默认是 `batch size×input size`

`weight ih`的形状是 `h dim×input size`

所以为了进行`batch`维度无关的乘法运算的话：

> 首先对`weight ih`进行一个扩充，把`weight`变成一个 `batch`的形式，`weight ih`是`hidden size×input size`的大小，对它 增加一维，`batch` 维度，对它进行复制，复制成跟`input`一样的大小，大小就变成了 `batch size×h dim×input size`

```python
w_ih_batch = weight_ih.unsqueeze(0).tile(bs,1,1) 
# bs*h_dim*input_size
```

这是 `w ih`，变成 `batch`的形状

同样对于`weight hh`，也是一样的，也转换一下，对它增加一个`batch`维度，然后把它的`batch`维度扩充成 `batch size`维度大小

```python
w_hh_batch = weight_hh.unsqueeze(0).title(bs,1,1)
# bs * h_dim * h_dim
```

这里 `w hh`大小就是 `batch size× h dim×h dim`，因为跟`hidden state`相连的，所以是一个方阵

$h_t = \mathrm{tanh(W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh})}$

第三步：开始计算 $w_{ih}× x_t、w_{hh}× h_{t-1}$

**第一项：`w_times_x`**

首先计算 `x`，就是计算 `Wih`乘以`x`这个量 `w_times_x`这个量，可以调用 `torch.bmm`这个函数

> `batch matrix multiplication`，是含有批大小的矩阵相乘，与 批 无关的 计算矩阵相乘

- 第一个位置传入 `w ih batch`
- 第二个位置 传入 `x`

> 当前这个`x`是`batch size× input_size`的，为了跟 `w ih batch`相乘，需要将它 扩充一维，扩充成 `batch size×input size×1`的，这里 需要 对它 扩充一下，调用一下`unsqueeze`
>
> ```python
> x = input[:,t,:].unsqueeze(2)
> ```

本来是二维的，现在在第三个维度上进行扩充，变成 `batch size×input size×1`，此时跟`x`相乘，得到 `batch size× h dim×1`，最后`1`的维度去掉，调用`unsqueeze`函数，得到的结果 `batch size×h dim`，得到`w times x`的结果，偏置最后再加

```python
x = input[:,t,:].unsqueeze(2)  # 获取当前时刻的输入特征 bs*input_size*1
w_ih_batch = weight_ih.unsqueeze(0).tile(bs,1,1) #bs*h_dim*input_size
w_hh_batch = weight_hh.unsqueeze(0).tile(bs,1,1) #bs*h_dim*h_dim

w_times_x = torch.bmm(w_ih_batch,x).squeeze(-1) # bs*h_dm
```

**第二项 `w_times_h`**

 `Whh` 矩阵 跟上一时刻的状态相乘的结果

同样调用 `torch.bmm`函数，带有批大小的矩阵相乘，跟上一时刻的隐含状态 进行相乘

同样对`h prev`进行扩充，`h_prev.unsqueeze(2)`，把它扩充三维

因为`h prev`本来是，`batch size×hidden size`，现在变成 `batch size×hidden size×1`，乘出来以后 是 `batch size×hidden size ×1`，最后再把1 去掉，挤掉

这里乘的权重是方阵，不改变大小，所以还是 `h prev`的形状

```python
w_times_h = torch.bmm(w_hh_batch,h_prev.unsqueeze(2)).squeeze(-1)
```

调用 `squeeze`函数，把最后的1去掉 最后变成了 `batch size× h dim`

这是这两个量，最后把这些东西全部加起来，跟`bias`加起来，然后通过两个tanh函数

![image-20241221195057262](images/image-20241221195057262.png)

首先是 `w_times_x`这个量 然后加上 `bias ih`，最后加上 `w times h`，上一时刻隐含状态相关的，最后是`bias hh`，然后过一个 `tanh`激活函数，最终得到当前时刻的这一状态

```python
torch.tanh(w_times_x + bias_ih + w_times_h + bias_hh)
```

定义为`h_prev`，因为进行的是递归的运算

```python
h_prev = torch.tanh(w_times_x + bias_ih + w_times_h + bias_hh)
```

现在计算了$t$时刻的输出，接着把$t$时刻的输出，放入到 `h out`中，

怎么放，只要放到时间长度这一维，$t$行即可

```python
h_out[:,t,:] = h_prev
```

以上完成了递归的运算，最后返回 跟 pytorch官方api一样

首先返回`h_out`

然后返回 最后一个时刻的隐含状态，其实也就是`h_prev`

但是这里的`h_prev`是二维的，官方api是三维的，所以要 扩一维，扩一维的原因就是因为 自己实现的是 单向、单层的，所以在 第0维 扩充一个1 就好了

```python
return h_out,h_prev.unsqueeze(0)
```

以上是所有全手写的RNN forward函数，其实就是单向的RNN

### torch.tile函数

补充 torch.tile函数：沿指定维度重复张量函数

例子：

```python
import torch

# 创建一个张量
weight_hh = torch.tensor([[1, 2], [3, 4]])

# 假设批量大小为3
bs = 3

# 使用 unsqueeze 在第0维度增加一个维度，然后使用 tile 沿第0维度重复 bs 次
w_hh_batch = weight_hh.unsqueeze(0).tile(bs, 1, 1)

print("原始张量:")
print(weight_hh)
print("增加维度并重复后的张量:")
print(w_hh_batch)
```

在这个示例中：

1. [`weight_hh`](vscode-file://vscode-app/Applications/Visual Studio Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 是一个形状为 `[2, 2]` 的张量。
2. [`weight_hh.unsqueeze(0)`](vscode-file://vscode-app/Applications/Visual Studio Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 在第0维度增加一个维度，使其形状变为 `[1, 2, 2]`。
3. [`tile(bs, 1, 1)`](vscode-file://vscode-app/Applications/Visual Studio Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 沿第0维度重复 [`bs`](vscode-file://vscode-app/Applications/Visual Studio Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 次（这里 [`bs`](vscode-file://vscode-app/Applications/Visual Studio Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 为3），使其形状变为 `[3, 2, 2]`。

输出结果：

```python
原始张量:
tensor([[1, 2],
        [3, 4]])
增加维度并重复后的张量:
tensor([[[1, 2],
         [3, 4]],

        [[1, 2],
         [3, 4]],

        [[1, 2],
         [3, 4]]])
```

这样，[`w_hh_batch`](vscode-file://vscode-app/Applications/Visual Studio Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 就是一个形状为 `[3, 2, 2]` 的张量，其中每个批次都包含原始的 [`weight_hh`](vscode-file://vscode-app/Applications/Visual Studio Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.esm.html) 张量

### 	5 验证

验证思路：

> 把之前实例化的RNN网络，参数拿出来，填充到自定义的网络中
>
> 然后算出来的结果 如果是跟官方API结果一致的话，就表明自定义的函数是正确的

**首先，拿出RNN的参数：**

（1）RNN有哪些参数？

`nn.Module`这个类，

① 在`pytorch`中 所有的层，都是继承自`nn.Module`这个类

② `nn.Module`的函数：`name.parameters`这个函数，查看 RNN中 有哪些参数

> `name.parameters`是一个生成器，可以用循环得到 `for p,n in` 
>
> p：参数
>
> n：name
>
> `in rnn.named_parameters():` 就能看到 rnn有哪些参数

```python
for p,n in rnn.named_parameters():
```

打印查看结果：RNN有哪些参数 以及 它的名称

![image-20241221200129294](images/image-20241221200129294.png)

可以看到 RNN的所有的参数、名称、具体地张量的数值 

一共有四个参数，分别是

①第一个参数： `weight ih l0`

- `weight ih` ： 公式里的 `wih`
- `l0`：网络定义只有一层，层数是从$0$开始的，所以是从`l0`

② 第二个参数：`weight hh l0`

表示当前层 `w hh`的参数

另外两个就是偏置了，分别是

③第三个参数： `bias ih`

④第四个参数： `bias hh`  

需要注意的是：

- 前面两个权重张量 是 二维张量
- 后面两个偏置是 一维的向量

（2）现在把这些参数 代入到自己写的`RNN forward`函数中

首先，复制一下 自己写的函数签名

```python
rnn_forward(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev):
```

- `input`还是`input`
- `weight ih`可以改成 `rnn.`，直接用`rnn.参数名称`，就可以访问这个参数 ，`rnn.weight_ih_l0`
- `weight hh`也是一样，用`rnn.`来进行访问：`rnn.weight_hh_l0`
- `bias`也是一样的 对应的是 `rnn.bias_ih_l0`
- 同样`hh bias`也是一样的 `rnn.bias_hh_l0`
- `h_prev`，就是自定义好的，就直接用`h_prev`

```python
rnn_forward(input,rnn.weight_ih_l0,rnn.weight_hh_l0,rnn.bias_ih_l0,rnn.bias_hh_l0,h_prev)
```

**变量名命令：**

前面写的是`rnn output`和`state finall`

![image-20241221201523163](images/image-20241221201523163.png)

加前缀 `custom`，表示自己写的

```python
custom_rnn_output,custom_state_finall = rnn_forward(input,rnn.weight_ih_l0,rnn.weight_hh_l0,rnn.bias_ih_l0,rnn.bias_hh_l0,h_prev)
```

这样就调用了自己写的`RNN forward`函数

然后对比`pytorch api`的结果 和 自己写的结果

![image-20241221201653310](images/image-20241221201653310.png)

第一个张量 整体RNN 预测的输出，是一致的

第二个张量是最后一个时刻的输出

官方的结果 和 自定义的结果一样 

#### 自定义 RNN代码

![image-20241221201752393](images/image-20241221201752393.png)

### 6 验证双向RNN

```python
h_t = tanh(x_t)
```

双向的话调用单向的函数

双向需要注意 所有的参数 都double了，所有的weight和bias 都有两个

```python
# step3 手写一个bidirectional_rnn_forward函数，实现双向RNN的计算原理
```

- 双向要考虑两倍的 `forward函数`和`backward层`
- `weight`有`forward`层和`backward`层
- `bias`也有`forward`层和`backward`层
- `h prev`也是有两份的

第一份是 `forward layer`，还有 `backward`，复制然后改名，按照官方的名称，改成`reverse`

这时候所有的参数都是两份的：

`forward`一份，`backward`一份

RNN是比较简单的，如果是`LSTM` 、`GRU` 更复杂

函数签名写成：

![image-20241221205304685](images/image-20241221205304685.png)

接下来，还是一样的，得到一些基本的信息

首先，上面复制下来：

![image-20241221205534221](images/image-20241221205534221.png)

第一步 `batch size`，`时间` 和 `input size`

然后，得到 `hidden size`  、`h_dim`

关于`h_out`，这里`batch size`不变，`T`不变，但是`h dim`要变成两倍，因为是双向的结构：

```python
h_out = torch.zeros(bs,T,h_dim*2)
# 初始化输出状态矩阵，注意双向是两倍的特征大小
```

该定义的定义好了，接下来 调用`RNN forward`函数

调用两次`RNN forward`函数

第一步一模一样

![image-20241221205736872](images/image-20241221205736872.png)

红框是需要传入的参数

这是`forward`层的调用，取名为 `forward_output`，这里只取 第一个返回值，所以加个[0]

![image-20241221205826587](images/image-20241221205826587.png)

得到 `forward layer`

下面 `backward layer`

这里要变换一下，除了所有的参数都用reverse版本的，对input 也要reverse一下，就是因为如果是反向的话，要保证第一个位置上，input是最后一个元素；对input 需要 在长度这一维 进行翻转：

![image-20241221205927963](images/image-20241221205927963.png)

调用 `torch.flip  api`，这个`api`，对张量进行翻转：

![image-20241221210032831](images/image-20241221210032831.png)

有两个参数：

- 一个是input

- 一个是dim，也就是说 传入的是 哪个 dim，就会对哪个 dim 进行翻转，完全相反的顺序

还是先拷贝所有的参数，调用 rnn_forward函数：

![image-20241221210122850](images/image-20241221210122850.png)

- 第一个参数 `input`，进行翻转，调用 `torch.flip`，`flip`的第一个参数是 `input`，第二个参数是`维度`，维度官方api中规定：

> ![image-20241221210222846](images/image-20241221210222846.png)
>
> 要么是列表 要么是元组

这里的input是三维，要翻转的是 中间这一维，`T`这维：

![image-20241221210258044](images/image-20241221210258044.png)

传入一个列表，`1`这一维度，表示中间这一维度，进行翻转

```python
rnn_forward(torch.flip(input,[1]),
            weight_ih_reverse,
            weight_hh_reverse,
            bias_ih_reverse,
            bias_hh_reveerse,
            h_prev_reverse)
```

同样 对它的调用 也只取 `output`，定义为 `backward output`

```python
backward_output = rnn_forward(torch.flip(input,[1]),
                              weight_ih_reverse,
                              weight_hh_reverse,
                              bias_ih_reverse,
                              bias_hh_reveerse,
                              h_prev_reverse)[0] # backward layer
```

以上 得到了 `forward output`和`backward output`

为什么 只保留了 `h_out`，没有保留`h prev`呢？

> 因为在RNN中，`h prev`可以从 `h out`中得到，所以为了方便 只取了 `h out`

![image-20241221210611796](images/image-20241221210611796.png)

接下来，把 `forward output` 和 `backward output` 填充到 `h out`中

首先 `h out`是三维的，并且最后一维 由 `forward 和 backward` 填充起来的，所以填充时，索引的写法：从$0$到 `h_dim`

```python
h_out[:,:,:h_dim] = forward_output
```

从`h_dim:`到最后

前向的输出，填充到前一半中，后一半的维度，用`backward output`填充

```python
h_out[:,:,h_dim:] = backward_output
```

把 `前向输出` 和 `后向输出` 拼起来，然后返回

同样按照`官方api`，返回两个数：

- 第一个数是 `h_out`
- 第二个数就是 `state finall`

![image-20241221210921980](images/image-20241221210921980.png)

`Sate finall`维度是 $D*num\_layers$ × N × $H_{out}$

- 前面表示 双向 和 层数的乘积
- 中间是`batch size`
- 后面是 `H_out`

怎么写呢？

首先 要取出  `h out`的最后一个时刻，因为时刻是在中间那个维度，所以用 `-1`索引

```python
return h_out,h_out[:,-1,:].reshape(())
```

先取出 最后一个时刻，最后一个时刻的状态向量，形状 $batch \_size×2倍的h\_dim$，先`reshape`，把2单独拎出来，然后reshape：

- Batch size不变
- 2单独拎出来
- h dim就写成 h dim

首先把二维张量 变成三维张量

```python
return h_out,h_out[:,-1,:].reshape((bs,2,h_dim))
```

然后 把2提到前面，根据官方api：

- 2 在前面

![image-20241221211444198](images/image-20241221211444198.png)

- batch size在中间

所以把2 提到前面，调用一下转置函数，就是把 `第0维度` 和 `第1维度` 交换一下：

```python
return h_out,h_out[:,-1,:].reshape((bs,2,h_dim)).transpose(0,1)
```

以上双向自定义RNN 函数的实现

#### 自定义双向 RNN代码

![image-20241221211612136](images/image-20241221211612136.png)

```python
# step3 手写一个 bidirectional_rnn_forward函数，实现双向RNN的计算原理
def bidirectional_rnn_forward(input,
                              weight_ih,
                              weight_hh,
                              bias_ih,
                              bias_hh,
                              h_prev,
                              weight_ih_reverse,
                              weight_hh_reverse,
                              bias_ih_reverse,
                              bias_hh_reverse,
                              h_prev_reverse):
    bs,T,input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs,T,h_dim*2) # 初始化一个输出（状态）矩阵，注意双向是两倍的特征大小

    forward_output = rnn_forward(input,
                                 weight_ih,
                                 weight_hh,
                                 bias_ih,
                                 bias_hh,
                                 h_prev)[0]  # forward layer
    backward_output = rnn_forward(torch.flip(input,[1]),
                                  weight_ih_reverse,
                                  weight_hh_reverse,
                                  bias_ih_reverse, 
                                  bias_hh_reverse,
                                  h_prev_reverse)[0] # backward layer

    # 将input按照时间的顺序翻转
    h_out[:,:,:h_dim] = forward_output
    h_out[:,:,h_dim:] = torch.flip(backward_output,[1]) #需要再翻转一下 才能和forward output拼接

    
    h_n = torch.zeros(bs,2,h_dim)  # 要最后的状态连接

    h_n[:,0,:] = forward_output[:,-1,:]
    h_n[:,1,:] = backward_output[:,-1,:]

    h_n = h_n.transpose(0,1)

    return h_out,h_n
    # return h_out,h_out[:,-1,:].reshape((bs,2,h_dim)).transpose(0,1)

# 验证一下 bidirectional_rnn_forward的正确性
bi_rnn = nn.RNN(input_size,
                hidden_size,
                batch_first=True,
                bidirectional=True)
h_prev = torch.zeros((2,bs,hidden_size))
bi_rnn_output,bi_state_finall = bi_rnn(input,h_prev)

for k,v in bi_rnn.named_parameters():
    print(k,v)
```

代码思路：

1. 首先把 `input`传入到 `forward layer`中
2. 然后再把`input` 按照 时间的顺序 翻转一下，再传入`backwardward layer`中
3. 再把 `forward output`和`backward output`拼起来，形成整体的`h out`
4. 最后返回序列 整体的隐含状态和 最后一个时刻的状态

现在验证  双向 rnn  forward 正确性

**首先 实例化双向RNN 层**

复制下来，并设置 bidirection=True

```python
# 验证一下 bidirectional_rnn_forward的正确性
bi_rnn = nn.RNN(input_size,hidden_size,batch_first=True,bidirectional=True)
```

同样定义一个`h_prev`

```python
h_prev = torch.zeros()
```

大小是 `2× batch size× hidden size`

![image-20241221212306336](images/image-20241221212306336.png)

```python
h_prev = torch.zeros(2,bs,hidden_size)
```

调用RNN，传入`input`和`h_prev`，得到双向RNN的`output`和双向`state finall`

```python
bi_rnn_output,bi_state_finall = bi_rnn(input,h_prev)
```

得到官方api的结果

![image-20241221212448882](images/image-20241221212448882.png)

对于RNN 查看一下 参数的名字，然后把这些参数代入到自定义的双向RNN函数中去

```python
for k,v in bi_rnn.named_parameters():
    print(k,v)
```

![image-20241221212548749](images/image-20241221212548749.png)

可以看到在pytorch双向RNN 中的参数：

1. weight ih l0
2. weight hh l0
3. bias ih l0
4. bias hh l0
5. weight ih l0 reverse
6. weight hh l0 reverse
7. bias ih l0
8. bias hh l0 reverse

一共有8个参数，这是因为 `forward layer`有4个参数，`reverse layer`也有4个参数

有了这8个参数，就可以把这8个参数传入到双向RNN中

首先把 签名 copy下来：

![image-20241221212723011](images/image-20241221212723011.png)

```python
bidirectional_rnn_forward(input,
                          weight_ih,
                          weight_hh,
                          bias_ih,
                          bias_hh,
                          h_prev,
                          weight_ih_reverse,
                          weight_hh_reverse,
                          bias_ih_reverse,
                          bias_hh_reverse,
                          h_prev_reverse)
```

- `input`不变
- `weight ih`改成`weight ih l0`
- `weight hh`，同样`weight hh l0`

还要加上`bi_rnn.`，也就是说把实例化的RNN层传进来

`bi_rnn.bias ih l0` 

`bi_rnn.bias hh l0`

![image-20241221212936350](images/image-20241221212936350.png)

`h prev`需要注意：是三维的

前面有个 2 ，只需要传入第一个就好了 `h prev[0]`

反向的也是类似的

`bi_rnn.weight ih l0 reverse`

后面也是一样 `bi_rnn.weight hh l0 reverse`

`bi_rnn.bias ih l0 reverse` 

`bi_rnn.bias hh l0 reverse`

`h prev reverse`，用`h prev [1]`

![image-20241221213102881](images/image-20241221213102881.png)

定义  `custom_bi_rnn_output,custom_bi_state_finall`接收输出

接下来分别打印api的结果 和 自己写的函数的结果：

![image-20241221213142401](images/image-20241221213142401.png)

> 这个 结果有问题，（后面改了 就是各种翻转 )
>

由于是双向的 `hidden size=3`，但是输出状态长度是6，这是因为双向的有拼接

## 汇总所有代码

```python
import torch
import torch.nn as nn
```

```python
bs,T=2,3  # 批大小，输入序列长度
input_size,hidden_size = 2,3 # 输入特征大小，隐含层特征大小
input = torch.randn(bs,T,input_size)  # 随机初始化一个输入特征序列
h_prev = torch.zeros(bs,hidden_size) # 初始隐含状态
```

```python
# step1 调用pytorch RNN API
rnn = nn.RNN(input_size,hidden_size,batch_first=True)
rnn_output,state_finall = rnn(input,h_prev.unsqueeze(0))

print(rnn_output)
print(state_finall)
```

输出：

```
tensor([[[-0.7709,  0.7301, -0.9299],
         [-0.6976, -0.8241, -0.1903],
         [-0.6485, -0.2633, -0.1093]],

        [[-0.2035,  0.7439, -0.1369],
         [-0.4805, -0.5790,  0.1787],
         [-0.6185,  0.4854, -0.4907]]], grad_fn=<TransposeBackward1>)
tensor([[[-0.6485, -0.2633, -0.1093],
         [-0.6185,  0.4854, -0.4907]]], grad_fn=<StackBackward0>)
```

```python
# step2 手写 rnn_forward函数，实现RNN的计算原理
def rnn_forward(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev):
    bs,T,input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs,T,h_dim) # 初始化一个输出（状态）矩阵

    for t in range(T):
        x = input[:,t,:].unsqueeze(2)  # 获取当前时刻的输入特征，bs*input_size*1
        w_ih_batch = weight_ih.unsqueeze(0).tile(bs,1,1) # bs * h_dim * input_size
        w_hh_batch = weight_hh.unsqueeze(0).tile(bs,1,1)# bs * h_dim * h_dim

        w_times_x = torch.bmm(w_ih_batch,x).squeeze(-1) # bs*h_dim
        w_times_h = torch.bmm(w_hh_batch,h_prev.unsqueeze(2)).squeeze(-1) # bs*h_him
        h_prev = torch.tanh(w_times_x + bias_ih + w_times_h + bias_hh)

        h_out[:,t,:] = h_prev

    return h_out,h_prev.unsqueeze(0)
```

```python
# 验证结果
custom_rnn_output,custom_state_finall = rnn_forward(input,
                                                    rnn.weight_ih_l0,
                                                    rnn.weight_hh_l0,
                                                    rnn.bias_ih_l0,
                                                    rnn.bias_hh_l0,
                                                    h_prev)
print(custom_rnn_output)
print(custom_state_finall)
```

输出：

```
tensor([[[-0.7709,  0.7301, -0.9299],
         [-0.6976, -0.8241, -0.1903],
         [-0.6485, -0.2633, -0.1093]],

        [[-0.2035,  0.7439, -0.1369],
         [-0.4805, -0.5790,  0.1787],
         [-0.6185,  0.4854, -0.4907]]], grad_fn=<CopySlices>)
tensor([[[-0.6485, -0.2633, -0.1093],
         [-0.6185,  0.4854, -0.4907]]], grad_fn=<UnsqueezeBackward0>)
```

```python
print(torch.allclose(rnn_output,custom_rnn_output))
print(torch.allclose(state_finall,custom_state_finall))
```

输出：True、True

```python
# step3 手写一个 bidirectional_rnn_forward函数，实现双向RNN的计算原理
def bidirectional_rnn_forward(input,
                              weight_ih,
                              weight_hh,
                              bias_ih,
                              bias_hh,
                              h_prev,
                              weight_ih_reverse,
                              weight_hh_reverse,
                              bias_ih_reverse,
                              bias_hh_reverse,
                              h_prev_reverse):
    bs,T,input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs,T,h_dim*2) # 初始化一个输出（状态）矩阵，注意双向是两倍的特征大小

    forward_output = rnn_forward(input,
                                 weight_ih,
                                 weight_hh,
                                 bias_ih,
                                 bias_hh,
                                 h_prev)[0]  # forward layer
    backward_output = rnn_forward(torch.flip(input,[1]),
                                  weight_ih_reverse,
                                  weight_hh_reverse,
                                  bias_ih_reverse, 
                                  bias_hh_reverse,
                                  h_prev_reverse)[0] # backward layer

    # 将input按照时间的顺序翻转
    h_out[:,:,:h_dim] = forward_output
    h_out[:,:,h_dim:] = torch.flip(backward_output,[1]) #需要再翻转一下 才能和forward output拼接

    
    h_n = torch.zeros(bs,2,h_dim)  # 要最后的状态连接

    h_n[:,0,:] = forward_output[:,-1,:]
    h_n[:,1,:] = backward_output[:,-1,:]

    h_n = h_n.transpose(0,1)

    return h_out,h_n
    # return h_out,h_out[:,-1,:].reshape((bs,2,h_dim)).transpose(0,1)

# 验证一下 bidirectional_rnn_forward的正确性
bi_rnn = nn.RNN(input_size,hidden_size,batch_first=True,bidirectional=True)
h_prev = torch.zeros((2,bs,hidden_size))
bi_rnn_output,bi_state_finall = bi_rnn(input,h_prev)

for k,v in bi_rnn.named_parameters():
    print(k,v)
```

输出

```
weight_ih_l0 Parameter containing:
tensor([[ 0.5458,  0.5512],
        [-0.5077, -0.0750],
        [ 0.3572,  0.1419]], requires_grad=True)
weight_hh_l0 Parameter containing:
tensor([[-0.4093,  0.2012,  0.0746],
        [-0.5619, -0.3820, -0.4060],
        [-0.4412,  0.2706, -0.2816]], requires_grad=True)
bias_ih_l0 Parameter containing:
tensor([-0.5063, -0.1391, -0.0587], requires_grad=True)
bias_hh_l0 Parameter containing:
tensor([ 0.0343, -0.2352,  0.3234], requires_grad=True)
weight_ih_l0_reverse Parameter containing:
tensor([[ 0.1298,  0.5538],
        [ 0.4151,  0.2533],
        [-0.4401,  0.5322]], requires_grad=True)
weight_hh_l0_reverse Parameter containing:
tensor([[-0.4232,  0.2246,  0.4265],
        [ 0.3016, -0.4142, -0.3064],
        [-0.1960,  0.2845,  0.3770]], requires_grad=True)
bias_ih_l0_reverse Parameter containing:
tensor([-0.4372, -0.2452,  0.4506], requires_grad=True)
bias_hh_l0_reverse Parameter containing:
tensor([ 0.3957, -0.4655, -0.2143], requires_grad=True)
```

```python
custom_bi_rnn_output,custom_bi_state_finall = bidirectional_rnn_forward(input,
                                                                        bi_rnn.weight_ih_l0,
                                                                        bi_rnn.weight_hh_l0,
                                                                        bi_rnn.bias_ih_l0,
                                                                        bi_rnn.bias_hh_l0,
                                                                        h_prev[0],
                                                                        bi_rnn.weight_ih_l0_reverse,
                                                                        bi_rnn.weight_hh_l0_reverse,
                                                                        bi_rnn.bias_ih_l0_reverse,
                                                                        bi_rnn.bias_hh_l0_reverse,
                                                                        h_prev[1])
```

```python
print("Pytorch API output")
print(bi_rnn_output)
print(bi_state_finall)

print("\n custom bidirectional_rnn_forward function output:")
print(custom_bi_rnn_output)
print(custom_bi_state_finall)
print(torch.allclose(bi_rnn_output,custom_bi_rnn_output))
print(torch.allclose(bi_state_finall,custom_bi_state_finall))
```

```
True
True
```

