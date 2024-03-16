#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

const int rows = 1000;
const int cols = 1000;

void initializeMatrix( vector<vector<double>>& matrix) {
    for (int i = 0; i < rows; ++i) {
        matrix.push_back(vector<double>(cols, 55.55));
    }
}

void matrixMultiplication(vector<vector<double>>& A, vector<vector<double>>& B, vector<vector<double>>& C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < cols; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void printMatrix(const vector<vector<double>>& matrix) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    vector<vector<double>> A, B, C;

    initializeMatrix(A);
    initializeMatrix(B);
    C.resize(rows, vector<double>(cols, 0));

    double start_time = omp_get_wtime();

    matrixMultiplication(A, B, C);

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    std::cout << "Matrix multiplication completed in " << elapsed_time << " seconds." << endl;
    // printMatrix(C);

    return 0;
}

// 25.6535 seconds.
