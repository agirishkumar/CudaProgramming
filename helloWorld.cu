#include <stdio.h>

// __global__ is a CUDA C keyword indicating that greetGpu() is a kernel function, which means it is a function that will be executed on the GPU.
__global__ void greetGpu(){
  printf("Hello World from the device: GPU\n");
  //prints the thread and block indices for the current execution. threadIdx.x gives the x-coordinate of the thread within the block, and blockIdx.x gives the x-coordinate of the block within the grid. This helps identify which thread and block are executing the code.
  printf("from the thread:[%d,%d]\n", threadIdx.x, blockIdx.x);
}

int main(){

  printf("Hello World from the host: CPU\n");
  // Calls the greetGpu kernel with a specific execution configuration. The triple angle brackets <<< >>> specify the execution configuration: the first number 2 indicates the number of blocks in the grid, and the second number 2 indicates the number of threads per block. This means the kernel will be executed by a total of 4 threads (2 blocks * 2 threads each).
  greetGpu<<<2,2>>>();
  // Waits for the completion of all preceding GPU tasks. Since kernel launches are asynchronous, cudaDeviceSynchronize ensures that the host (CPU) waits until the GPU has completed all tasks, including printing to the console, before proceeding.
  // cudaDeviceSynchronize();
  return 0;
}