import sys
INT_MIN = -sys.maxsize-1

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ret = INT_MIN
        for i in range(len(nums)):
            tmp = 0
            for j in range(i, len(nums)):                
                tmp += nums[j]
                ret = max(ret, tmp)
        return ret

