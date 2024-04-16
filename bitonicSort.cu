#include<iostream>

__global__ void bitonic_sort_step(int *dev_values, int j, int k){
  unsigned int i, ixj;
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i ^ j;

  if ((ixj) > i) {
        if ((i & k) == 0) {
            if (dev_values[i] > dev_values[ixj]) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        else if (dev_values[i] < dev_values[ixj]) {
            int temp = dev_values[i];
            dev_values[i] = dev_values[ixj];
            dev_values[ixj] = temp;
        }
    }

}

void bitonic_sort(int *values, int N) {
    int *dev_values;
    size_t size = N * sizeof(int);

    cudaMalloc((void**) &dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks(N / 256, 1);  // Adjust this as necessary
    dim3 threads(256, 1);

    // Major step
    for (int k = 2; k <= N; k <<= 1) {
        // Sub-step
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}

int main() {
    int N = 1024;  // Number of elements; N should be a power of 2
    int values[N];

    // Initialize the array with random numbers
    for (int i = 0; i < N; i++) {
        values[i] = rand() % 100;
    }

    bitonic_sort(values, N);

    // Output the sorted array
    for (int i = 0; i < N; i++) {
        std::cout << values[i] << " ";
    }
    printf("\n");

    return 0;
}