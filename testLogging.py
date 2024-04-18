import logging
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)

def my_function():
    # Simulating some task within the loop
    time.sleep(0.1)

# Before optimization
start_time = time.time()
for _ in range(10):
    my_function()
end_time = time.time()
logging.info(f"Execution time before optimization: {end_time - start_time} seconds")

# After optimization
start_time = time.time()
with ThreadPoolExecutor() as executor:
    executor.map(my_function, range(10))
end_time = time.time()
logging.info(f"Execution time after optimization: {end_time - start_time} seconds")
