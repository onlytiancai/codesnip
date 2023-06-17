# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        current = head
        tail = None
        while current is not None:            
            next = current.next
            current.next = tail
            tail = current

            if next is None:
                return current
            current = next
