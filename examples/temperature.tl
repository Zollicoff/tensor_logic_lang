// Temperature Sigmoid for Embedding Space Reasoning
//
// From the paper: "σ(x,T) = 1/(1+e^(-x/T))"
// - T=0: Pure deductive reasoning (hard Boolean)
// - T→∞: Analogical reasoning (soft similarities)
//
// This enables "sound reasoning in embedding space" - combining
// neural network scalability with symbolic reasoning reliability.

domain n: 4
domain d: 3

// Embeddings for entities
Emb[n,d] = 0
Emb[0,0] = 1.0   // Entity 0: [1, 0, 0]
Emb[1,1] = 1.0   // Entity 1: [0, 1, 0]
Emb[2,2] = 1.0   // Entity 2: [0, 0, 1]
Emb[3,0] = 0.7   // Entity 3: [0.7, 0.7, 0] - similar to 0 and 1
Emb[3,1] = 0.7

// Relation embedding (learned)
RelEmb[d,d] = 0
RelEmb[0,0] = 2.0
RelEmb[1,1] = 2.0
RelEmb[2,2] = 2.0

// Compute similarity scores
// Score[i,j] = Emb[i] @ RelEmb @ Emb[j]^T
Score[i,j] = Emb[i,d] RelEmb[d,d'] Emb[j,d']

// Low temperature (T=0.1): Near-Boolean, sharp decisions
// High scores → 1, low scores → 0
Sharp[i,j] = sigmoid(Score[i,j], 0.1)

// Medium temperature (T=1.0): Balanced soft reasoning
Medium[i,j] = sigmoid(Score[i,j], 1.0)

// High temperature (T=5.0): Very soft, analogical
// Even moderate scores get significant probability
Soft[i,j] = sigmoid(Score[i,j], 5.0)

// Query results to see the difference
Score?
Sharp?
Medium?
Soft?
