// Test: Matrix multiplication with consistent contracted index
// j is contracted (appears on RHS but not LHS)
// j must have consistent domain across A and B

domain I: 3
domain J: 4
domain K: 2

// Declare A and B with explicit shapes
sparse A[I, J]
sparse B[J, K]

// Initialize
A[i,j] = 1
B[j,k] = 2

// Matrix multiply - j is contracted
// j appears at A dim1 (domain J, size 4) and B dim0 (domain J, size 4) - consistent!
C[i,k] = A[i,j] B[j,k]
