class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        ret = -1
        curr = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                curr += 1
                ret = max(curr, ret)
            else:
                curr = 0     
        return max(ret, curr)
