class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class ListNode2:
    def __init__(self, x):
        self.val = x
        self.pre = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        t = head 
        last = None
        while t:
            t2 = ListNode2(t)
            t2.pre = last 
            last = t2
            t = t.next
        print('--- iter recv')
        t = last
        while t and k:
            print(t.val)
            t = t.pre
            k = k-1
        return t.val


if __name__ == '__main__':
    n = None
    h = None
    for i in range(1,7):
        t = ListNode(i)
        if n:
            n.next = t
        n = t 
        if not h:
            h = n

    print('--- iter input')
    t = h 
    while t:
        print(t.val)
        t = t.next

    s = Solution()
    h = s.getKthFromEnd(h, 2)
    print('--- iter result')
    t = h 
    while t:
        print(t.val)
        t = t.next
