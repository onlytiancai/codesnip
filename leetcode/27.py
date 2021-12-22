class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        n = len(nums)
        count = 0
        swap_count = 0
        for i in range(n):
            if nums[i] == val:
                if i < n - swap_count:
                    count = count + 1
                for j in range(n-1, i, -1):
                    if nums[j] != val:
                        swap_count += 1
                        t = nums[j]
                        nums[j] = nums[i]
                        nums[i] = t
                        break                    
        return n - count
