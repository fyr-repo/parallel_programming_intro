#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int SIZE = 1000;

// Function to initialize a matrix with a given value
void initializeMatrix(double* matrix, double value) {
    for (int i = 0; i < SIZE * SIZE; i++) {
        matrix[i] = value;
    }
}

// Function to multiply two matrices
void matrixMultiplication(double* mat1, double* mat2, double* result) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; k++) {
                sum += mat1[i * SIZE + k] * mat2[k * SIZE + j];
            }
            result[i * SIZE + j] = sum;
        }
    }
}

// Function to display a matrix
void displayMatrix(double* matrix) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << fixed << setprecision(2) << matrix[i * SIZE + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    // Allocate memory for matrices on the heap
    double* matrix1 = new double[SIZE * SIZE];
    double* matrix2 = new double[SIZE * SIZE];
    double* result = new double[SIZE * SIZE];

    // Initialize both matrices with the value 55.55
    initializeMatrix(matrix1, 55.55);
    initializeMatrix(matrix2, 55.55);

    // Start measuring time
    auto start = high_resolution_clock::now();

    // Perform matrix multiplication
    matrixMultiplication(matrix1, matrix2, result);

    // Stop measuring time
    auto stop = high_resolution_clock::now();

    // Calculate the elapsed time
    auto duration = duration_cast<milliseconds>(stop - start);

    // Display the elapsed time
    cout << "Time Elapsed for Matrix Multiplication: " << duration.count() << " milliseconds" << endl;

    // Display a portion of the resulting matrix
    cout << "Resultant Matrix (displaying first 10x10 elements):" << endl;
    displayMatrix(result);

    // Deallocate memory
    delete[] matrix1;
    delete[] matrix2;
    delete[] result;

    return 0;
}

// 5.763 seconds