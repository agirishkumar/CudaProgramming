#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMul(const int *A, const int *B, int *C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

void initializeMatrix(int *mat, int rows, int cols) {
    int value = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = ++value; // Sequential integers
        }
    }
}

int main() {
    int m = 512, n = 512, p = 512;
    size_t sizeA = m * n * sizeof(int);
    size_t sizeB = n * p * sizeof(int);
    size_t sizeC = m * p * sizeof(int);

    int *h_A = new int[m * n];
    int *h_B = new int[n * p];
    int *h_C = new int[m * p];
    
    // Initialize matrices
    initializeMatrix(h_A, m, n);
    initializeMatrix(h_B, n, p);

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "CUDA execution time: " << milliseconds << " ms\n";

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Display the first element for verification
    std::cout << "First element of result (CUDA): " << h_C[0] << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
