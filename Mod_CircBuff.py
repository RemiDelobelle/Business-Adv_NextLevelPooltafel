class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.idx = 0  

    def add(self, item):
        self.buffer[self.idx] = item
        self.idx = (self.idx + 1) % self.size  

    def get(self):
        return self.buffer
    
    def exists_within_margin(self, center, margin):
        for item in self.buffer:
            if item is not None and abs(center[0] - item[0]) < margin and abs(center[1] - item[1]) < margin:
                return True
        return False