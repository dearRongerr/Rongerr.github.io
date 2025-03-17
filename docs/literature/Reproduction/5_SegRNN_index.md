# SegRNN

- [x] 可逆实例标准化 对输入序列进行标准化

```
x = self.revinLayer(x, 'norm').permute(0, 2, 1)
```

对预测序列进行反标准化

```
y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
```

- [x] vscode 禁用断点的方法

![image-20250315163423821](images/image-20250315163423821.png)
