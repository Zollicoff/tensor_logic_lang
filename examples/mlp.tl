// Multi-Layer Perceptron
//
// A 2-layer neural network: input -> hidden -> output
// Demonstrates: matmul, relu, softmax

domain b: 2
domain i: 3
domain h: 4
domain o: 2

// Input: 2 samples, 3 features
X[b,i] = 0
X[0,0] = 1.0
X[0,1] = 0.5
X[0,2] = -0.5
X[1,0] = -1.0
X[1,1] = 2.0
X[1,2] = 0.0

// Layer 1 weights: 4x3
W1[h,i] = 0
W1[0,0] = 0.1
W1[0,1] = 0.2
W1[0,2] = 0.3
W1[1,0] = -0.1
W1[1,1] = 0.4
W1[1,2] = 0.1
W1[2,0] = 0.2
W1[2,1] = -0.2
W1[2,2] = 0.5
W1[3,0] = 0.3
W1[3,1] = 0.1
W1[3,2] = -0.1

// Layer 2 weights: 2x4
W2[o,h] = 0
W2[0,0] = 0.5
W2[0,1] = -0.3
W2[0,2] = 0.2
W2[0,3] = 0.1
W2[1,0] = -0.4
W2[1,1] = 0.6
W2[1,2] = -0.1
W2[1,3] = 0.3

// Forward pass
Z1[b,h] = W1[h,i] X[b,i]
H[b,h] = relu(Z1[b,h])

Z2[b,o] = W2[o,h] H[b,h]
Y[b,o] = softmax(Z2[b,o])
