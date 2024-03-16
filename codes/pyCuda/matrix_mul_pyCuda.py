# ! nvcc --version

# ! pip install nvcc4jupyter
# ! pip install pycuda

# %load_ext nvcc4jupyter

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


from pycuda import gpuarray, tools, compiler

import time

kernel_code_template = """
__global__ void matrixmulti(int matrixsize,float *a, float *b, float *c)
{

    // 2D Thread ID 
    int tx = blockDim.x*blockIdx.x + threadIdx.x; // Compute column index
    int ty = blockDim.y*blockIdx.y + threadIdx.y; // Compute row index

    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    if((ty <matrixsize) && (tx < matrixsize))
    {
    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;
    for(int k=0; k<matrixsize;++k)
    {
    float Aelement = a[ty*matrixsize +k];
    float Belement = b[k*matrixsize +tx];
    Pvalue += Aelement * Belement;
    }
    c[ty * matrixsize + tx] = Pvalue;
    }

}
"""

# Define the dimensions and values of the arrays
rows_cols = 1000
value = 55.55

BLOCK_SIZE = 32

# Create the input arrays filled with the specified value
array1 = np.full((rows_cols, rows_cols), value).astype(np.float32)
array2 = np.full((rows_cols, rows_cols), value).astype(np.float32)



array1_gpu = gpuarray.to_gpu(array1) 
array2_gpu = gpuarray.to_gpu(array2)

result_gpu = gpuarray.empty((rows_cols, rows_cols), np.float32)



# compile the kernel code
mod = compiler.SourceModule(kernel_code_template)

# get the kernel function from the compiled module
matrixmul = mod.get_function("matrixmulti")

MATRIX_SIZE = rows_cols

# set grid size
if MATRIX_SIZE%BLOCK_SIZE != 0:
    grid=(MATRIX_SIZE//BLOCK_SIZE+1,MATRIX_SIZE//BLOCK_SIZE+1,1)
else:
    grid=(MATRIX_SIZE//BLOCK_SIZE,MATRIX_SIZE//BLOCK_SIZE,1)

matrixsize=MATRIX_SIZE

# Measure the start time
start_time = time.time()

# call the kernel on the card
matrixmul(np.uint32(matrixsize),
    # inputs
    array1_gpu, array2_gpu,
    # output
    
    result_gpu,
    grid=grid,
    block = (BLOCK_SIZE, BLOCK_SIZE, 1),
    )

# Measure the end time
end_time = time.time()

# Calculate the time taken for processing
processing_time = end_time - start_time

# Print the result and time taken for processing
print("Matrix multiplication result:")
print(result_gpu.get())
print(f"Time taken for processing: {processing_time:.4f} seconds")

# np.allclose(result_gpu, result_gpu.get())

# 0.0039 seconds