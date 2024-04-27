import random

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.idx = 0  # Index to keep track of the current position in the buffer

    def add(self, item):
        self.buffer[self.idx] = item
        self.idx = (self.idx + 1) % self.size  # Move to the next position, wrapping around if needed

    def get(self):
        return self.buffer

# Create a circular buffer to store the last 20 ball locations
buffer_size = 20
ball_buffer = CircularBuffer(buffer_size)

# Simulate detecting balls and adding their locations to the buffer
for _ in range(50):  # Simulate 50 frames
    # Generate a random ball location (x, y)
    ball_location = (random.randint(0, 100), random.randint(0, 100))
    print("Detected ball at location:", ball_location)
    
    # Add the ball location to the circular buffer
    ball_buffer.add(ball_location)

# Retrieve and print the last 20 ball locations from the buffer
last_20_balls = ball_buffer.get()
print("\nLast 20 ball locations:")
for idx, location in enumerate(last_20_balls):
    if location is not None:
        print(f"Index {idx}: {location}")
