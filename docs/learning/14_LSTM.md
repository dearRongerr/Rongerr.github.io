# LSTM

## Recall

![image-20241221142624552](images/image-20241221142624552.png)

RNNCELL，可以理解为 单步 的迭代

因为所有的循环神经网络 都是有很多步 去迭代，最终把每一步的状态 取出来 作为 输出

这里的RNNCELL，也就是说 多个，每个时刻的计算 就是一个RNNCELL，然后把 多个RNNCELL 连起来 ，其实就构成了 一个RNN，所以 无论是RNN也好，还是 GRU也好，还是 LSTM也好，它们 都有各自的CELL，然后每个CELL，其实就是一个 单步的运算，可以理解为 单个时刻的运算，下面 有一个例子

![image-20241221142949694](images/image-20241221142949694.png)

可以看到，首先 实例化了 一个 RNNCELL；这个RNNCELL的 input size和hidden size分别为10和20；然后 我们 定义一个 input 的训练特征，batch size是3，然后 时间长度是6，然后特征维度是10，并且定义了 一个 初始的 hidden state[hx],然后就可以用RNNCELL，来去做每一次迭代，所以我们 看到 这里有一个 for循环，然后 每一步 会调用 这个RNNCELL的 实例化的操作，算出每一时刻的隐含状态 hx=rnn(input[i],hx)  这儿应该是 $h_x$

