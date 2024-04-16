#include <iostream>
#include <cstdlib>

// CUDA Kernel function to multiply the elements of two arrays
__global__ void multiply(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] * b[index];
    }
}

void random_ints(int* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = rand() % 100;
    }
}

int main() {
    int n = 1024; // number of elements in each array
    int *a, *b, *c;           // host copies of a, b, c
    int *d_a, *d_b, *d_c;     // device copies of a, b, c
    int size = n * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Setup input values
    a = (int *)malloc(size); random_ints(a, n);
    b = (int *)malloc(size); random_ints(b, n);
    c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch multiply() kernel on GPU with enough blocks
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    multiply<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Display the results
    for(int i = 0; i < 10; i++) {
        std::cout << a[i] << "*" << b[i] << "=" << c[i] << std::endl;
    }

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
