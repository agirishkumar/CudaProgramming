#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

int main() {
    // Load image using OpenCV
    cv::Mat img = cv::imread("input.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully!" << std::endl;
    std::cout << "Image dimensions: " << img.cols << " x " << img.rows << std::endl;

    return 0;
}
