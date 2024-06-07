#include <iostream>
#include <math.h>

#define N 2048

using namespace std;

// Kernel function to multiply vector elements by a scalar
__global__ void vectorScalarMul(int *d_vec, int scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_vec[idx] *= scalar;
    }
}

int main() {

    int size = N * sizeof(int);

    // Allocate host and device memory
    int *h_vec = (int *)malloc(size);
    int *d_vec;
    cudaMalloc((void **)&d_vec, size);

    // Initialize random seed
    srand(time(0));

    // Initialize vector on the host
    for (int i = 0; i < N; ++i) {
        h_vec[i] = rand() % 100;
    }

    // Copy vector from host to device
    cudaMemcpy(d_vec, h_vec, size, cudaMemcpyHostToDevice);

    // Set scalar value
    int scalar = 8;

    // Launch the kernel
    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorScalarMul<<<numBlocks, blockSize>>>(d_vec, scalar, N);

    // Copy result back from device to host
    cudaMemcpy(h_vec, d_vec, size, cudaMemcpyDeviceToHost);

    // Print the first 10 elements of the result
    cout << "First 10 elements of the result:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << h_vec[i] << " ";
    }
    cout << endl;

    // Free memory
    free(h_vec);
    cudaFree(d_vec);

    return 0;
}