// Training with Autodiff
//
// Linear regression with gradient descent
// Demonstrates: forward pass, loss computation, backward pass

domain n: 2
domain f: 4
domain o: 1

// Weights (to be learned)
W[f,o] = 0.5
b[o] = 0.1

// Training data
X[n,f] = 0
X[0,0] = 1.0
X[0,1] = 0.5
X[0,2] = 0.2
X[0,3] = 0.1
X[1,0] = 0.3
X[1,1] = 0.8
X[1,2] = 0.4
X[1,3] = 0.2

// Target values
Target[n,o] = 0
Target[0,0] = 3.35
Target[1,0] = 2.7

// Forward pass: Y = X @ W + b
Linear[n,o] = X[n,f] W[f,o]
Y[n,o] = Linear[n,o] + b[o]

// Loss: mean squared error
Diff[n,o] = Y[n,o] - Target[n,o]
Loss[n,o] = Diff[n,o] * Diff[n,o]

// Backward pass: compute gradients dL/dW, dL/db
backward Loss wrt W, b
