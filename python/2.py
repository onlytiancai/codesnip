nums = [4, 3, 2, 7, 8, 2, 3, 1]
expect = [5, 6]
actual = []

for i in range(len(nums)):
    if nums[abs(nums[i])-1] > 0:
        nums[abs(nums[i])-1] = -nums[abs(nums[i])-1] 

for i in range(len(nums)):
    if nums[i] > 0:
        actual.append(i+1)

print(nums, expect, actual)
