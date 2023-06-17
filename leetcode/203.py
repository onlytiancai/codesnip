# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        if head is None:
            return head     
        
        ret = ListNode(None, None)
        prev = ret
        current = head        
        while current is not None:                                
            if current.val != val:                
                prev.next = current
                prev = current
            current = current.next
        if prev.next is not None and prev.next.val == val:
            prev.next = None  
        return ret.next
