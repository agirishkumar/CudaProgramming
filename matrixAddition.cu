#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

// Error checking function (same as before)
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void matrixAdd(const int *d_A, const int *d_B, int *d_C, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m*n){
        d_C[i] = d_A[i] + d_B[i]; 
    }    
}

void fillMatrix(int *h_A, int m, int n){
    for (int i = 0; i < m * n; ++i)
    {
        h_A[i] = rand() % 100;
    }
}

int main(){

    // dimensions of the matrix
    int m = 1024;
    int n = 1024;
    int size = m*n*sizeof(int); 

    // Allocate memory for the matrix vectors on host
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    // Initialize random seed
    srand(time(NULL));

    // fill the matrices with elements
    fillMatrix(h_A, m, n);
    fillMatrix(h_B, m, n);
   
    // create matrix vectors on the device
    int *d_A, *d_B, *d_C;

    // Allocate memory for the matrix vectors on the device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    checkCUDAError("cudaMalloc failed");

    // copy the host matrices to the device
    cudaMemcpy( d_A, h_A ,size , cudaMemcpyHostToDevice);
    cudaMemcpy( d_B, h_B ,size , cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy (host to device) failed");

    // launching the kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (m*n + threadsPerBlock -1)/threadsPerBlock;
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,m,n);
    checkCUDAError("Kernel launch failed");

    // copy the device matrix result to host
    cudaMemcpy( h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy (device to host) failed");

    // Print the first 10 elements of the result
    cout << "First 10 elements of the result matrix: " << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << h_C[i] <<" ";
    }
    cout << endl;
    
    // free the memory on host and device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}