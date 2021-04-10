import sys
INT_MIN = -sys.maxsize-1

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ret = INT_MIN
        sum = 0
        for i in range(len(nums)):
            sum += nums[i]
            ret = max(sum, ret)            
            if sum < 0:
                sum = 0
        return ret
