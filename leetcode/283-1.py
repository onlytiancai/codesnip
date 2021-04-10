class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = 0
        while i < n:
            if nums[i] == 0:
                j = i
                while j < n:
                    if nums[j] != 0:
                        t = nums[j]
                        nums[j] = nums[i]
                        nums[i] = t
                        break
                    j = j + 1
            i = i + 1
