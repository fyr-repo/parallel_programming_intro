import numpy as np
import time
import multiprocessing

def multiply_row_col(args):
    row, col, A, B = args
    return np.dot(A[row], B[:, col])

def parallel_matrix_multiplication(A, B):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    rows, cols = A.shape[0], B.shape[1]
    args_list = [(i, j, A, B) for i in range(rows) for j in range(cols)]
    result = pool.map(multiply_row_col, args_list)
    pool.close()
    pool.join()
    return np.array(result).reshape(rows, cols)

if __name__ == "__main__":
    # Define the dimensions and values of the arrays
    rows_cols = 1000
    value = 55.55

    # Create the input arrays filled with the specified value
    array1 = np.full((rows_cols, rows_cols), value).astype(np.float32)
    array2 = np.full((rows_cols, rows_cols), value).astype(np.float32)

    # Measure the start time
    start_time = time.time()

    result = parallel_matrix_multiplication(array1, array2)
    
    # Measure the end time
    end_time = time.time()

    # Calculate the time taken for processing
    processing_time = end_time - start_time

    # Print the result and time taken for processing
    print("Matrix multiplication result:")
    print(result)
    print(f"Time taken for processing: {processing_time:.4f} seconds")

# 6.6914 seconds