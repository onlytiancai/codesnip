import sys
INT_MIN = -sys.maxsize-1

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ret = INT_MIN
        dp = nums[0]
        for i in range(1, len(nums)):
            ret = max(dp, ret)
            dp = max(dp + nums[i], nums[i])            
        ret = max(ret, dp)            
        return ret
