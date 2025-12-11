// Self-Attention
//
// Core mechanism of transformers: Attention(Q,K,V) = softmax(QK^T) V
// Demonstrates: einsum, softmax, attention patterns

domain s: 4
domain d: 3

// Query vectors
Q[q,d] = 0
Q[0,0] = 1.0
Q[0,2] = 1.0
Q[1,0] = 0.5
Q[1,1] = 0.5
Q[2,1] = 1.0
Q[2,2] = 0.5
Q[3,0] = 1.0
Q[3,1] = 1.0
Q[3,2] = 1.0

// Key vectors
K[k,d] = 0
K[0,0] = 1.0
K[1,1] = 1.0
K[2,2] = 1.0
K[3,0] = 0.5
K[3,1] = 0.5
K[3,2] = 0.5

// Value vectors
V[k,d] = 0
V[0,0] = 1.0
V[1,1] = 1.0
V[2,2] = 1.0
V[3,0] = 1.0
V[3,1] = 1.0
V[3,2] = 1.0

// Attention scores: Q @ K^T (dot product over d)
Scores[q,k] = Q[q,d] K[k,d]

// Softmax over keys (each query attends to all keys)
Attn[q,k] = softmax(Scores[q,k])

// Weighted sum of values
Out[q,d] = Attn[q,k] V[k,d]
