import sys
INT_MIN = -sys.maxsize-1

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ret = INT_MIN
        dp = None
        for i in range(len(nums)):
            dp = max((0 if dp is None else dp) + nums[i], nums[i])
            ret = max(dp, ret)
        return ret
