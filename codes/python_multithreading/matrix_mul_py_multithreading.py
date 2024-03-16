import numpy as np
import time
import threading

# Define the dimensions and values of the arrays
rows_cols = 1000
value = 55.55

# Create the input arrays filled with the specified value
array1 = np.full((rows_cols, rows_cols), value).astype(np.float32)
array2 = np.full((rows_cols, rows_cols), value).astype(np.float32)

# Function to perform matrix multiplication for a specific block
def multiply_block(start_row, end_row, start_col, end_col, result):
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            result[i, j] = np.sum(array1[i, :] * array2[:, j])

# Create a shared result array
result = np.zeros((rows_cols, rows_cols))

# Define the number of threads and the block size
num_threads = 4
block_size = rows_cols // num_threads

# Measure the start time
start_time = time.time()

# Create threads to perform matrix multiplication for each block
threads = []
for i in range(num_threads):
    start_row = i * block_size
    end_row = (i + 1) * block_size if i < num_threads - 1 else rows_cols
    thread = threading.Thread(target=multiply_block, args=(start_row, end_row, 0, rows_cols, result))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Measure the end time
end_time = time.time()

# Calculate the time taken for processing
processing_time = end_time - start_time

# Print the result and time taken for processing
print("Matrix multiplication result:")
print(result)
print(f"Time taken for processing: {processing_time:.4f} seconds")

# 30.9589 seconds