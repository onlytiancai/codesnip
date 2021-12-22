class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.l = []


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.l.append(x)


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.l.pop()


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.l[len(self.l) - 1]


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self.l) == 0



# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
