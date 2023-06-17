# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        ret = ListNode()
        current = ret
        jinwei = 0
        while l1 or l2:            
            total = (l1.val if l1 else 0) + (l2.val if l2 else 0) + jinwei
            current.next = ListNode(total % 10)            
            jinwei = total // 10;
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
            current = current.next
        if jinwei:
            current.next = ListNode(jinwei)
        return ret.next
