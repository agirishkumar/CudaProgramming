#include <iostream>
#include <cuda_runtime.h>

__global__ void rgb2gray(unsigned char* d_in, unsigned char* d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = d_in[idx];
        unsigned char g = d_in[idx + 1];
        unsigned char b = d_in[idx + 2];
        d_out[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Simplified input image (3x3 RGB image)
    int width = 3;
    int height = 3;
    unsigned char h_in[27] = { // 3x3 image with RGB values
        255, 0, 0,   0, 255, 0,   0, 0, 255,
        255, 255, 0, 0, 255, 255, 255, 0, 255,
        128, 128, 128, 64, 64, 64, 32, 32, 32
    };
    unsigned char h_out[9]; // 3x3 grayscale image

    int imgSize = width * height * 3;
    int grayImgSize = width * height;

    // Allocate device memory
    unsigned char* d_in;
    unsigned char* d_out;
    checkCudaError(cudaMalloc((void**)&d_in, imgSize), "Failed to allocate device memory for input image");
    checkCudaError(cudaMalloc((void**)&d_out, grayImgSize), "Failed to allocate device memory for output image");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_in, h_in, imgSize, cudaMemcpyHostToDevice), "Failed to copy input image from host to device");

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    rgb2gray<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Copy the result back to the host
    checkCudaError(cudaMemcpy(h_out, d_out, grayImgSize, cudaMemcpyDeviceToHost), "Failed to copy output image from device to host");

    // Print the result
    std::cout << "Grayscale image:" << std::endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << (int)h_out[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
