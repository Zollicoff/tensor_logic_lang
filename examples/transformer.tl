// Transformer Self-Attention in Tensor Logic
//
// Simplified transformer encoder demonstrating:
// 1. Linear projections (Q, K, V)
// 2. Self-attention with softmax
// 3. Feed-forward network with ReLU
//
// Based on "Attention is All You Need" (Vaswani et al., 2017)

// Dimensions
domain i: 4
domain j: 4
domain k: 4
domain d: 4

// Input embeddings X[i, k] - 4 tokens, 4-dim embeddings
X[i,k] = 0
X[0,0] = 1.0
X[1,1] = 1.0
X[2,2] = 1.0
X[3,3] = 1.0

// Weight matrices (simplified)
Wq[k,d] = 0
Wk[k,d] = 0
Wv[k,d] = 0

// Identity-like projections
Wq[0,0] = 1.0
Wq[1,1] = 1.0
Wq[2,2] = 1.0
Wq[3,3] = 1.0
Wk[0,0] = 1.0
Wk[1,1] = 1.0
Wk[2,2] = 1.0
Wk[3,3] = 1.0
Wv[0,0] = 1.0
Wv[1,1] = 1.0
Wv[2,2] = 1.0
Wv[3,3] = 1.0

// Project to Q, K, V
Q[i,d] = X[i,k] Wq[k,d]
K[j,d] = X[j,k] Wk[k,d]
V[j,d] = X[j,k] Wv[k,d]

// Attention scores: Q @ K^T
Scores[i,j] = Q[i,d] K[j,d]

// Softmax attention weights
Attn[i,j] = softmax(Scores[i,j])

// Weighted sum of values
AttnOut[i,d] = Attn[i,j] V[j,d]

// Feed-forward: ReLU(X @ W1) @ W2
W1[k,d] = 0
W2[d,k] = 0
W1[0,0] = 0.5
W1[1,1] = 0.5
W1[2,2] = 0.5
W1[3,3] = 0.5
W2[0,0] = 0.5
W2[1,1] = 0.5
W2[2,2] = 0.5
W2[3,3] = 0.5

Hidden[i,d] = AttnOut[i,k] W1[k,d]
HiddenRelu[i,d] = relu(Hidden[i,d])
Output[i,k] = HiddenRelu[i,d] W2[d,k]
