import tensorflow as tf
import numpy as np
import time

# Define the dimensions and values of the arrays
rows_cols = 1000
value = 55.55

# Create the input arrays filled with the specified value
array1 = tf.constant([[value] * rows_cols] * rows_cols, dtype=tf.float32)
array2 = tf.constant([[value] * rows_cols] * rows_cols, dtype=tf.float32)

# Measure the start time
start_time = time.time()

# Perform matrix multiplication
result = tf.matmul(array1, array2)

# Measure the end time
end_time = time.time()

# Calculate the time taken for processing
processing_time = end_time - start_time

# Print the result and time taken for processing
print("Matrix multiplication result:")
print(result)
print(f"Time taken for processing: {processing_time:.4f} seconds")

# 0.0427 seconds
