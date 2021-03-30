class Solution:
    def twoSum(self, nums, target):
        hash_table = {}
        for i, x in enumerate(nums):
            diff = target - x
            print(i, x, diff)
            if diff in hash_table:
                diff_index = hash_table[diff]
                if diff_index != i:
                    return [diff_index, i]
            else:
                hash_table[x] = i
        return []

s = Solution()
r = s.twoSum([2,7,11,15], 9)
print(r)
