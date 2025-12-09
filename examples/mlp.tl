// Multi-Layer Perceptron in Tensor Logic
//
// A simple 2-layer MLP: input -> hidden -> output
// h = relu(W1 @ x + b1)
// y = softmax(W2 @ h + b2)
//
// In tensor notation:
//   Hidden[b,h] = relu(W1[h,i] Input[b,i])
//   Output[b,o] = softmax(W2[o,h] Hidden[b,h])

// Dimensions - use index names as domain names
domain b: 2
domain i: 3
domain h: 4
domain o: 2

// Input data: 2 samples, 3 features each
Input[b,i] = 0
Input[0,0] = 1.0
Input[0,1] = 0.5
Input[0,2] = -0.5
Input[1,0] = -1.0
Input[1,1] = 2.0
Input[1,2] = 0.0

// Layer 1 weights: hidden x input (4x3)
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

// Layer 2 weights: output x hidden (2x4)
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

// Layer 1: Z1 = W1 @ Input, then apply ReLU
Z1[b,h] = W1[h,i] Input[b,i]
Hidden[b,h] = relu(Z1[b,h])

// Layer 2: Z2 = W2 @ Hidden, then apply softmax
Z2[b,o] = W2[o,h] Hidden[b,h]
Probs[b,o] = softmax(Z2[b,o])
