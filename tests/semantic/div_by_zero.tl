// Test: Division by zero in index should produce error

domain n: 10

A[i] = 1

// Division by zero - ERROR
B[i/0] = A[i]
