class Solution(object):
    def searchRange(self,nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        def lfunc(nums,target):
            n = len(nums)
            left = 0
            right = n - 1
            while(left<=right):
                mid = (left+right)//2
                if nums[mid] >= target:
                    right = mid-1
                else:
                    left = mid+1
            return left
        def rfunc(nums,target):
            n = len(nums)-1
            left = 0
            right = n 
            while(left<=right):
                mid = (left+right)//2
                if nums[mid] > target:
                    right = mid-1
                else:
                    left = mid+1
            return right

        a = lfunc(nums,target)
        b = rfunc(nums,target)
        if  a == len(nums) or nums[a] != target:
            return [-1,-1]
        else:
            return [a,b]


nums = [1]
target = 1
a = Solution().searchRange(nums, target)