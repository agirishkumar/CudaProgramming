#include<iostream>

__global__ void matrixMul(int *A, int *B, int *C, int m, int n, int p){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col <p){
    int sum =0;
    for (int k=0; k<n; k++){
      sum += A[row*n+k]*B[k*p+col];
    }
    C[row*p + col] = sum; 
  }
}

void fillMatrix(int *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = rand() % 10;  // Random integers between 0 and 9
    }
}

int main(){
  int m = 1024;
  int n = 1024;
  int p = 1024;

  int *h_A = (int *)malloc(m*n*sizeof(int));
  int *h_B = (int *)malloc(n*p*sizeof(int));
  int *h_C = (int *)malloc(m*p*sizeof(int));

  fillMatrix(h_A, m, n);
  fillMatrix(h_B, n, p);

  int *d_A, *d_B, *d_C;

  cudaMalloc((void **)&d_A, m*n*sizeof(int));
  cudaMalloc((void **)&d_B, n*p*sizeof(int));
  cudaMalloc((void **)&d_C, m*p*sizeof(int));

  cudaMemcpy(d_A, h_A, m*n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, n*p*sizeof(int), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);

  cudaMemcpy(h_C, d_C, m*p*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      std::cout << h_C[i*p+j] << " ";
    }
    std::cout << std::endl;
  }

  free(h_A);
  free(h_B);
  free(h_C);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}