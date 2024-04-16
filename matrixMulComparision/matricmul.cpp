#include <iostream>
#include <vector>
#include <chrono>

void initializeMatrix(std::vector<std::vector<int>>& mat) {
    int value = 0;
    for (auto &row : mat) {
        for (auto &cell : row) {
            cell = ++value; // Sequential integers
        }
    }
}

void multiplyMatrices(const std::vector<std::vector<int>>& A,
                      const std::vector<std::vector<int>>& B,
                      std::vector<std::vector<int>>& C,
                      int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int m = 512, n = 512, p = 512;
    std::vector<std::vector<int>> A(m, std::vector<int>(n));
    std::vector<std::vector<int>> B(n, std::vector<int>(p));
    std::vector<std::vector<int>> C(m, std::vector<int>(p));

    initializeMatrix(A);
    initializeMatrix(B);

    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatrices(A, B, C, m, n, p);
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> milliseconds = stop - start;
    std::cout << "C++ execution time: " << milliseconds.count() << " ms\n";
    std::cout << "First element of result (C++): " << C[0][0] << std::endl;

    return 0;
}
