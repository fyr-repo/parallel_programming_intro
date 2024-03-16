#include <iostream>
#include <vector>
#include <mpi.h>

const int rows = 1000;
const int cols = 1000;

void initializeMatrix(std::vector<std::vector<double>>& matrix) {
    for (int i = 0; i < rows; ++i) {
        matrix.push_back(std::vector<double>(cols, 55.55));
    }
}

void matrixMultiplication(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rowsPerProcess = rows / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank + 1) * rowsPerProcess;

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < cols; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int start = p * rowsPerProcess;
            int end = (p + 1) * rowsPerProcess;
            std::vector<double> tempRow(cols);

            MPI_Recv(&tempRow[0], cols * rowsPerProcess, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = start; i < end; ++i) {
                for (int j = 0; j < cols; ++j) {
                    C[i][j] = tempRow[j];
                }
            }
        }
    } else {
        MPI_Send(&C[startRow][0], cols * rowsPerProcess, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::vector<double>> A, B, C;

    
    initializeMatrix(A);
    initializeMatrix(B);
    C.resize(rows, std::vector<double>(cols, 0));
    

    double start_time = MPI_Wtime();

    matrixMultiplication(A, B, C);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    
    std::cout << "Matrix multiplication completed in " << elapsed_time << " seconds." << std::endl;
    std::cout << "Matrix C:" << std::endl;
    // printMatrix(C);
    

    MPI_Finalize();

    return 0;
}

// mpic++ matrix_mul_cpp_mpi.cpp
// 26.6837 seconds.
