// Matrix Multiplication
//
// The simplest tensor equation: C = A @ B
// Repeated index j is summed (Einstein notation)

domain I: 3
domain J: 4
domain K: 2

// Matrix A: 3x4
A[i,j] = 0
A[0,0] = 1.0
A[0,1] = 2.0
A[0,2] = 3.0
A[0,3] = 4.0
A[1,0] = 5.0
A[1,1] = 6.0
A[1,2] = 7.0
A[1,3] = 8.0
A[2,0] = 9.0
A[2,1] = 10.0
A[2,2] = 11.0
A[2,3] = 12.0

// Matrix B: 4x2
B[j,k] = 0
B[0,0] = 1.0
B[0,1] = 2.0
B[1,0] = 3.0
B[1,1] = 4.0
B[2,0] = 5.0
B[2,1] = 6.0
B[3,0] = 7.0
B[3,1] = 8.0

// Matrix multiply: C[i,k] = sum_j A[i,j] * B[j,k]
C[i,k] = A[i,j] B[j,k]
