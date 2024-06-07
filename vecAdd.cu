#define N 512

int main(){

    int *a, *b, *c; // host copies of a,b,c
    int *d_a, *d_b, *d_c; // device copies of a,b,c
    int size = N * sizeof(int);

    // Allocate space for device copies of a,b,c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for host copies of a,b,c
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    for (int i =0; i< N; i++){
        
    }


    return 0;
}