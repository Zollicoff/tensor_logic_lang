// Training Example for Tensor Logic
// Demonstrates gradient-based learning using differentiable tensor operations
//
// This example shows how to train a simple linear regression model
// to learn the weights that minimize squared error.

// Define index domains
domain i: 4     // input features
domain j: 1     // output (scalar)
domain n: 10    // number of training samples

// Model parameters (to be learned)
// Weights: W[i,j] maps inputs to outputs
W[i,j] = 0.5

// Bias term
b[j] = 0.1

// Training data (simple linear relationship: y = 2*x1 + 3*x2 - 1*x3 + 0.5*x4)
// Sample 1
X[0,0] = 1.0
X[0,1] = 0.5
X[0,2] = 0.2
X[0,3] = 0.1
Y_target[0] = 2.0 * 1.0 + 3.0 * 0.5 - 1.0 * 0.2 + 0.5 * 0.1  // = 3.35

// Sample 2
X[1,0] = 0.3
X[1,1] = 0.8
X[1,2] = 0.4
X[1,3] = 0.2
Y_target[1] = 2.0 * 0.3 + 3.0 * 0.8 - 1.0 * 0.4 + 0.5 * 0.2  // = 2.7

// Forward pass: Y_pred = X @ W + b
// Using Einstein notation: Y_pred[n] = X[n,i] W[i,j] + b[j]
Y_pred[n,j] = X[n,i] W[i,j] + b[j]

// Loss: Mean Squared Error
// MSE = mean((Y_pred - Y_target)^2)
diff[n,j] = Y_pred[n,j] - Y_target[n]
loss[n,j] = diff[n,j] * diff[n,j]

// Gradient: d_loss/d_W = X^T @ (Y_pred - Y_target) * 2/n
// Using autodiff, gradients are computed automatically

// Print results
// After running with --trace, you'll see intermediate values
// The optimizer module provides SGD, Momentum, and Adam to update W and b
