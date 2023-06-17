class MyCircularQueue:

    def __init__(self, k: int):
        self.capacity = k
        self.list = list(range(self.capacity))
        self.head = 0
        self.tail = 0
        self.count = 0
        

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False

        if self.head >= self.capacity:
            self.head = self.head - self.capacity

        self.list[self.head] = value
        self.head += 1
        self.count += 1
        return True


    def deQueue(self) -> bool:
        if self.isEmpty():
            return False

        if self.tail >= self.capacity:
            self.tail -= self.capacity
            
        self.tail += 1
        self.count -= 1
        return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        if self.tail >= self.capacity:
            self.tail -= self.capacity            
        return self.list[self.tail]

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
            
        return self.list[self.head-1]


    def isEmpty(self) -> bool:
        return self.count == 0

    def isFull(self) -> bool:
        return self.count == self.capacity



# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
