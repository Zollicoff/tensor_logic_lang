// Test: Index used with different domain sizes - should ERROR
// Same index 'i' is used in two tensors with different domain sizes

domain a: 3
domain b: 5

sparse A[a]
sparse B[b]

// i appears in A (domain a, size 3) and B (domain b, size 5)
// This is inconsistent - same index symbol used with different sizes
C[i] = A[i] B[i]
