class RecentCounter:

    def __init__(self):
        self.q = []

    def ping(self, t: int) -> int:
        self.q.append(t)        
        n = len(self.q)
        while n > 0:            
            if self.q[n-1] < t - 3000:
                return len(self.q) - n
            n = n - 1
        return len(self.q)

# Your RecentCounter object will be instantiated and called as such:
# obj = RecentCounter()
# param_1 = obj.ping(t)
