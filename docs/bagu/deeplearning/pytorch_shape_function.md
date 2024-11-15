# pytorch的维度变换公式
1. `torch.unsqueeze(input, dim)`：在指定维度 `dim` 上增加一个新的维度。如果 `dim` 已经存在，则在其前面添加新的维度。
2. `torch.squeeze(input, dim=None)`：移除所有长度为1的维度。如果指定了 `dim`，则只移除该维度。
3. `torch.flatten(input, start_dim=0, end_dim=-1)`：将输入张量从 `start_dim` 到 `end_dim` 的所有维度展平。
4. `torch.view(input, size)` 或 `input.view(size)`：重新调整张量的形状，不改变数据。
5. `torch.reshape(input, shape)`：与 `view` 类似，用于改变张量的形状，但 `reshape` 可以处理更复杂的维度变换，如增加或减少维度。
6. `torch.permute(input, dims)`：重新排列输入张量的维度，`dims` 是一个维度索引的元组。
7. `torch.transpose(input, dim0, dim1)`：交换输入张量的两个维度。
8. `torch.expand(input, size)`：将输入张量沿指定的维度复制扩展。
9. `torch.cat(tensors, dim)`：沿指定维度 `dim` 连接多个张量。
10. `torch.stack(tensors, dim)`：沿新的维度 `dim` 堆叠多个张量，与 `cat` 不同的是，`stack` 会增加一个新的维度。
11. `torch.reapeat`