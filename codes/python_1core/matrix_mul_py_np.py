import numpy as np
import time

def matrix_multiplication(matrix1, matrix2):
    result = np.dot(matrix1, matrix2)
    return result


# Define the dimensions and values of the arrays
rows_cols = 1000
value = 55.55

# Create the input arrays filled with the specified value
array1 = np.full((rows_cols, rows_cols), value)
array2 = np.full((rows_cols, rows_cols), value)

# Measure the start time
start_time = time.time()

result = matrix_multiplication(array1, array2)

# Measure the end time
end_time = time.time()

# Calculate the time taken for processing
processing_time = end_time - start_time

# Print the result and time taken for processing
print("Matrix multiplication result:")
print(result)
print(f"Time taken for processing: {processing_time:.4f} seconds")

# 0.0403 seconds