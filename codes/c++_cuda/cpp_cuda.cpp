// ! nvcc --version

// ! pip install nvcc4jupyter

// %load_ext nvcc4jupyter

// %%cuda

#include <iostream>
#include <ctime>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int i = 0; i < colsA; ++i) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// Function to initialize a matrix 
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = 55.55;
    }
}

int main() {
    
    clock_t start = clock();
            
    const int rowsA = 1000;
    const int colsA = 1000;
    const int rowsB = colsA;
    const int colsB = 1000;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = rowsB * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);

    // Allocate memory for host matrices
    h_A = (float*)malloc(sizeA);
    h_B = (float*)malloc(sizeB);
    h_C = (float*)malloc(sizeC);

    // Initialize host matrices
    initializeMatrix(h_A, rowsA, colsA);
    initializeMatrix(h_B, rowsB, colsB);

    // Allocate memory for device matrices
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 64);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);

    clock_t end = clock();

    double duration = double(end - start) / CLOCKS_PER_SEC;

    // Print a few elements of the result matrix
    std::cout << "Result of matrix multiplication:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << h_C[i * colsB + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Time taken: " << duration << " seconds" << std::endl;

    return 0;
}

// 0.00021 seconds
