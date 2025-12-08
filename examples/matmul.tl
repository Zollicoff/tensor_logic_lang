// Matrix multiplication as a tensor equation
// C[i,k] = sum_j A[i,j] * B[j,k]

// In tensor logic, adjacent tensors multiply and shared indices sum
C[i,k] = A[i,j] B[j,k]
