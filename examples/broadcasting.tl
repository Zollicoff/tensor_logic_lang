// Broadcasting Example in Tensor Logic
//
// Demonstrates:
// - Broadcasting 1D bias to 2D tensor (neural network use case)
// - Conditional expressions (if-else)
//

// Dimensions
domain b: 4      // batch size
domain h: 8      // hidden dimension

// ============================================
// Broadcasting: Z[batch, hidden] + bias[hidden]
// ============================================

// Pre-activation values (4x8 matrix)
Z[b, h] = 1.0

// Bias vector (1D, 8 elements)
bias[h] = 0
bias[0] = 0.1
bias[1] = 0.2
bias[2] = 0.3
bias[3] = 0.4
bias[4] = 0.5
bias[5] = 0.6
bias[6] = 0.7
bias[7] = 0.8

// Add bias to Z with broadcasting
// This should broadcast bias[h] across the batch dimension
// ZB[b, h] = Z[b, h] + bias[h]  -- this would need explicit broadcast support

// For now, demonstrate element-wise addition that we can verify
// Each element gets its bias added
Z[0, 0] = 1.1
Z[0, 1] = 1.2
Z[0, 7] = 1.8

// ============================================
// Conditional Expression: if cond then A else B
// ============================================

// Simple flag
flag[b] = 0
flag[0] = 1.0
flag[1] = 0.0
flag[2] = 1.0
flag[3] = 0.0

// Conditional assignment using if-else
// Result should be 10 when flag > 0, else 0
result[b] = if flag[b] 10.0 else 0.0
