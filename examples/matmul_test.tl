// Matrix multiplication test with explicit initialization
// C[i,k] = A[i,j] B[j,k]
// Since we don't have literal tensor initialization yet,
// we test the einsum machinery with domains

domain i: 2
domain j: 3
domain k: 2

// Declare tensors
A[i,j] = 0
B[j,k] = 0

// The actual multiplication (will be zeros since inputs are zeros)
C[i,k] = A[i,j] B[j,k]
