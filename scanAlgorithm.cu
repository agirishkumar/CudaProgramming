#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Kernel declarations
__global__ void upsweep(int* d, int n, int stride);
__global__ void downsweep(int* d, int n, int stride);

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::vector<int> scan(const std::vector<int>& inp) {
    int n = inp.size();
    int* d;
    size_t size = n * sizeof(int);

    // Allocate memory on device
    checkCudaError(cudaMalloc((void**)&d, size), "Failed allocating memory on device");

    // Transfer data from host to device
    checkCudaError(cudaMemcpy(d, inp.data(), size, cudaMemcpyHostToDevice), "Failed transferring data from host to device!");

    int blockSize = 256;
    int gridSize;

    // Upsweep phase
    for (int stride = 1; stride < n; stride *= 2) {
        gridSize = (n / (stride * 2) + blockSize - 1) / blockSize;
        gridSize = max(gridSize, 1); // Ensure grid size is at least 1
        std::cout << "Upsweep: gridSize=" << gridSize << ", blockSize=" << blockSize << ", stride=" << stride << std::endl;
        upsweep<<<gridSize, blockSize>>>(d, n, stride);
        checkCudaError(cudaGetLastError(), "Kernel launch failed in upsweep");
    }

    // Set last element to zero before down-sweep
    checkCudaError(cudaMemset(&d[n - 1], 0, sizeof(int)), "Failed to set last element to zero");

    // Downsweep phase
    for (int stride = n / 2; stride > 0; stride /= 2) {
        gridSize = (n / (stride * 2) + blockSize - 1) / blockSize;
        gridSize = max(gridSize, 1); // Ensure grid size is at least 1
        std::cout << "Downsweep: gridSize=" << gridSize << ", blockSize=" << blockSize << ", stride=" << stride << std::endl;
        downsweep<<<gridSize, blockSize>>>(d, n, stride);
        checkCudaError(cudaGetLastError(), "Kernel launch failed in downsweep");
    }

    // Transfer the data from device to host
    std::vector<int> output(n);
    checkCudaError(cudaMemcpy(output.data(), d, size, cudaMemcpyDeviceToHost), "Failed to copy data from device to host");

    // Free device memory
    checkCudaError(cudaFree(d), "Failed to free device memory");

    return output;
}

__global__ void upsweep(int* d, int n, int stride) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;
    if (index + stride < n) {
        d[index + stride * 2 - 1] += d[index + stride - 1];
    }
}

__global__ void downsweep(int* d, int n, int stride) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;
    if (index + stride < n) {
        int temp = d[index + stride - 1];
        d[index + stride - 1] = d[index + stride * 2 - 1];
        d[index + stride * 2 - 1] += temp;
    }
}

void print(const std::vector<int>& out) {
    for (int i : out) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> inp = {10, 20, 15, 8, 45, 6};
    std::vector<int> out = scan(inp);

    print(out);

    return 0;
}
