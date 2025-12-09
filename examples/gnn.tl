// Graph Neural Network in Tensor Logic
//
// Implements message passing on a graph:
// 1. Message: aggregate neighbor features
// 2. Update: combine with self features
//
// This is similar to Graph Convolutional Networks (GCN)
// and can express transitive closure like Datalog

// Graph with 5 nodes, 4-dimensional features
domain n: 5
domain m: 5
domain f: 4

// Adjacency matrix A[n,m] - 1 if edge from n to m
// A simple chain graph: 0 -> 1 -> 2 -> 3 -> 4
A[n,m] = 0
A[0,1] = 1
A[1,2] = 1
A[2,3] = 1
A[3,4] = 1

// Add self-loops (common in GCN)
A[0,0] = 1
A[1,1] = 1
A[2,2] = 1
A[3,3] = 1
A[4,4] = 1

// Node features X[n,f]
X[n,f] = 0
// Node 0: [1, 0, 0, 0]
X[0,0] = 1.0
// Node 1: [0, 1, 0, 0]
X[1,1] = 1.0
// Node 2: [0, 0, 1, 0]
X[2,2] = 1.0
// Node 3: [0, 0, 0, 1]
X[3,3] = 1.0
// Node 4: [1, 1, 1, 1]
X[4,0] = 1.0
X[4,1] = 1.0
X[4,2] = 1.0
X[4,3] = 1.0

// Layer 1: Message Passing
// H1[n,f] = sum_m A[n,m] * X[m,f]
// (aggregate neighbor features weighted by adjacency)
H1[n,f] = A[n,m] X[m,f]

// Apply nonlinearity
H1_relu[n,f] = relu(H1[n,f])

// Layer 2: Another round of message passing
H2[n,f] = A[n,m] H1_relu[m,f]
H2_relu[n,f] = relu(H2[n,f])

// Layer 3: Final aggregation with sigmoid
H3[n,f] = A[n,m] H2_relu[m,f]
Output[n,f] = sigmoid(H3[n,f])

// == Reachability as GNN ==
// The Ancestor relation can be seen as a GNN without features:
// Reachable[i,k] = A[i,k] | exists j: A[i,j] & Reachable[j,k]
//
// In tensor logic with fixpoint iteration:
// Reach[i,k] max= step(A[i,k])
// Reach[i,k] max= step(A[i,j] Reach[j,k])

// Initialize Reachability from adjacency
Reach[n,m] = 0
Reach[n,m] max= step(A[n,m])
// Transitive closure (run with -f flag for fixpoint)
Reach[n,m] max= step(A[n,k] Reach[k,m])
