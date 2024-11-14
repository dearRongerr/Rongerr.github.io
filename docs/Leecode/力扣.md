## 两数之和

力扣1 两数之和 

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

```

**示例 1：**

```python
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
```

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict = {}

        for i in range(len(nums)):
            if target - nums[i] not in dict:
                dict[nums[i]] = i
            else:
                return [dict[target-nums[i]],i]
```