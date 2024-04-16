#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMulKernel(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void runExperiment(int N, dim3 blockSizes) {
    int *A, *B, *C;
    size_t size = N * N * sizeof(int);
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);
    // Initialize A, B here

    dim3 gridSize((N + blockSizes.x - 1) / blockSizes.x, (N + blockSizes.y - 1) / blockSizes.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulKernel<<<gridSize, blockSizes>>>(A, B, C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Execution time for block size " << blockSizes.x << "x" << blockSizes.y << ": " << milliseconds << " ms\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

int main() {
    int N = 16384;  // Size of the matrix
    runExperiment(N, dim3(16, 16));
    runExperiment(N, dim3(32, 32));
    runExperiment(N, dim3(64, 64));
    runExperiment(N, dim3(128, 128));
    runExperiment(N, dim3(256, 256));
    runExperiment(N, dim3(512, 512));
    runExperiment(N, dim3(1024, 1024));
    runExperiment(N, dim3(2048, 2048));
    runExperiment(N, dim3(4096, 4096));
    runExperiment(N, dim3(8192, 8192));
    runExperiment(N, dim3(16384, 16384));
    

    return 0;
}
