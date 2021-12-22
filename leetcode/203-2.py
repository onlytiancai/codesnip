# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        ret = ListNode(None, head)
        prev = ret      
        while prev.next is not None:                                
            if prev.next.val == val:                
                prev.next = prev.next.next
            else:
                prev = prev.next
        return ret.next
