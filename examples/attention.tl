// Self-Attention in Tensor Logic
//
// From Pedro Domingos' paper: Attention is just einsum + softmax
// Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
//
// In tensor notation:
//   Scores[i,j] = Q[i,k] K[j,k]        (dot products between queries and keys)
//   Attn[i,j] = softmax(Scores[i,j])   (over j dimension)
//   Output[i,d] = Attn[i,j] V[j,d]     (weighted combination of values)

// Dimensions: 4 tokens, 3-dimensional embeddings
// Use index names as domain names so they map correctly
domain i: 4
domain j: 4
domain k: 3
domain d: 3

// Initialize tensors to zero
Q[i,k] = 0
K[j,k] = 0
V[j,d] = 0

// Q[i,k] - query for each position i
Q[0,0] = 1.0
Q[0,1] = 0.0
Q[0,2] = 1.0
Q[1,0] = 0.5
Q[1,1] = 0.5
Q[1,2] = 0.0
Q[2,0] = 0.0
Q[2,1] = 1.0
Q[2,2] = 0.5
Q[3,0] = 1.0
Q[3,1] = 1.0
Q[3,2] = 1.0

// K[j,k] - key for each position j
K[0,0] = 1.0
K[0,1] = 0.0
K[0,2] = 0.0
K[1,0] = 0.0
K[1,1] = 1.0
K[1,2] = 0.0
K[2,0] = 0.0
K[2,1] = 0.0
K[2,2] = 1.0
K[3,0] = 0.5
K[3,1] = 0.5
K[3,2] = 0.5

// V[j,d] - value for each position j
V[0,0] = 1.0
V[0,1] = 0.0
V[0,2] = 0.0
V[1,0] = 0.0
V[1,1] = 1.0
V[1,2] = 0.0
V[2,0] = 0.0
V[2,1] = 0.0
V[2,2] = 1.0
V[3,0] = 1.0
V[3,1] = 1.0
V[3,2] = 1.0

// Step 1: Compute attention scores (dot product of Q and K)
// Scores[i,j] = sum_k Q[i,k] * K[j,k]
Scores[i,j] = Q[i,k] K[j,k]

// Step 2: Apply softmax to get attention weights
// This normalizes each row to sum to 1
Attn[i,j] = softmax(Scores[i,j])

// Step 3: Compute weighted sum of values
// Output[i,d] = sum_j Attn[i,j] * V[j,d]
Output[i,d] = Attn[i,j] V[j,d]
