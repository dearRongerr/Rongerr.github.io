# 预训练权重 

![image-20250221104534879](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504201219833.png)

![image-20250221104549576](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504201219834.png)

![image-20250221104631382](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504201219835.png)

![image-20250221104819303](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504201219836.png)

问题在于不能一一对应：

`load_state_dict` 中的 `strict = True`

![image-20250221104957505](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504201219837.png)

## 解决办法1

![image-20250221105131601](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504201219838.png)

`torch.load_state_dict`，加参数，`strict=False`



## 解决方法 2

加判断语句

![image-20250221105656618](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504201219839.png)

`model_dict`  是自己改的

`ckpt` 原始的

- 在哪儿改？

![image-20250221110820428](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504201219840.png)

找到源码中 加载预训练权重的地方 ，修改为上面的判断语句代码

## 单 GPU 和多 GPU

![image-20250221111510757](https://cdn.jsdelivr.net/gh/dearRongerr/PicGo@main/202504201219841.png) 

 把 `module.` 替换成 `''` 空，保证名字是一致的，可以加载预训练权重。