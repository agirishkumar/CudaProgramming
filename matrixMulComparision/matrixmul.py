import time

def initialize_matrix(rows, cols):
    """Initialize the matrix with sequential integers starting from 1."""
    return [[i * cols + j + 1 for j in range(cols)] for i in range(rows)]

def multiply_matrices(A, B):
    """Multiply two matrices using standard triple-loop method."""
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    return C

m, n, p = 512, 512, 512
A = initialize_matrix(m, n)
B = initialize_matrix(n, p)

start_time = time.time()
C = multiply_matrices(A, B)
end_time = time.time()

print("Python execution time: {:.2f} ms".format((end_time - start_time) * 1000))
print("First element of result (Python):", C[0][0])
