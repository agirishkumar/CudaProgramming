#include <iostream>
#include <cstdlib>
#include <ctime>

#define M 1024
#define N 1024
#define K 1024

using namespace std;

// Kernel function to multiply two matrices using 1D indexing
__global__ void matrixMultiply1D(const int *d_A, const int *d_B, int *d_C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        int value = 0;
        for (int e = 0; e < n; ++e) {
            value += d_A[row * n + e] * d_B[e * k + col];
        }
        d_C[row * k + col] = value;
    }
}

// Function to fill a matrix with random values
void fillMatrix(int *h_A, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = rand() % 100; // Random values between 0 and 99
    }
}

int main() {
    // Dimensions of the matrices
    int m = M;
    int n = N;
    int k = K;
    int size_A = m * n * sizeof(int);
    int size_B = n * k * sizeof(int);
    int size_C = m * k * sizeof(int);

    // Allocate memory for the matrices on host
    int *h_A = (int *)malloc(size_A);
    int *h_B = (int *)malloc(size_B);
    int *h_C = (int *)malloc(size_C);

    // Initialize random seed
    srand(time(0));

    // Fill the matrices with random elements
    fillMatrix(h_A, m, n);
    fillMatrix(h_B, n, k);

    // Create matrix vectors on the device
    int *d_A, *d_B, *d_C;

    // Allocate memory for the matrices on the device
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // Copy the host matrices to the device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Launching the kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (m * k + threadsPerBlock - 1) / threadsPerBlock;
    matrixMultiply1D<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print the first 10 elements of the result matrix for verification
    cout << "First 10 elements of the result matrix:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << h_C[i] << " ";
    }
    cout << endl;

    // Free the device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
