// Test: Tensor rank mismatch should produce error
// A is first defined as rank 2, then used as rank 1

domain n: 10

// First use: A is rank 2
A[i,j] = 1

// Second use: A is rank 1 - ERROR: rank mismatch
B[i] = A[i]
