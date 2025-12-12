// Test: Consistent tensor ranks should pass
// All uses of each tensor have the same rank

domain n: 10
domain m: 5

// A is always rank 2
A[i,j] = 1
B[i,j] = A[i,j] + 1

// C is always rank 1 (j is contracted)
C[i] = A[i,j]

// D is always rank 2
D[i,k] = A[i,j] B[j,k]
