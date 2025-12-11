// Graph Neural Network
//
// Message passing on a graph: aggregate neighbor features
// Also shows transitive closure (reachability)

domain n: 5
domain f: 4

// Adjacency matrix (chain: 0->1->2->3->4 with self-loops)
A[n,m] = 0
A[0,0] = 1
A[0,1] = 1
A[1,1] = 1
A[1,2] = 1
A[2,2] = 1
A[2,3] = 1
A[3,3] = 1
A[3,4] = 1
A[4,4] = 1

// Node features (one-hot + node 4 is all ones)
X[n,f] = 0
X[0,0] = 1.0
X[1,1] = 1.0
X[2,2] = 1.0
X[3,3] = 1.0
X[4,0] = 1.0
X[4,1] = 1.0
X[4,2] = 1.0
X[4,3] = 1.0

// Layer 1: aggregate neighbor features
H1[n,f] = A[n,m] X[m,f]
H1r[n,f] = relu(H1[n,f])

// Layer 2: another round
H2[n,f] = A[n,m] H1r[m,f]
H2r[n,f] = relu(H2[n,f])

// Layer 3: final with sigmoid
H3[n,f] = A[n,m] H2r[m,f]
Out[n,f] = sigmoid(H3[n,f])

// Reachability (transitive closure)
Reach[n,m] = step(A[n,m])
Reach[n,m] max= step(A[n,k] Reach[k,m])
