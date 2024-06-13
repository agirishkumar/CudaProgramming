#include <iostream>
#include <opencv2/opencv.hpp>
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
    // Load image using OpenCV
    cv::Mat img = cv::imread("power.jpeg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int imgSize = width * height * img.channels();
    int grayImgSize = width * height;  // Define grayImgSize here

    // Allocate host memory
    unsigned char* h_in = img.data;
    unsigned char* h_out = new unsigned char[grayImgSize];

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

    // Create output image and save it
    cv::Mat grayImg(height, width, CV_8UC1, h_out);
    cv::imwrite("output.jpg", grayImg);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Free host memory
    delete[] h_out;

    return 0;
}
