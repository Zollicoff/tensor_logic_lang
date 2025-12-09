// Simple 2-Layer Neural Network in Tensor Logic
//
// Demonstrates forward pass through a small neural network:
// - Input: 4 features
// - Hidden layer: 8 neurons with ReLU activation
// - Output: 2 classes with softmax
//
// This example shows the forward computation.
// Training would use autodiff (see src/runtime/autodiff.zig)

// Dimensions
domain b: 4      // 4 samples in mini-batch
domain i: 4      // 4 input features
domain h: 8      // 8 hidden neurons
domain o: 2      // 2 output classes

// ============================================
// Input data X[b, i]
// ============================================
X[b, i] = 0

// Sample 1: [1, 0, 0, 0]
X[0, 0] = 1.0

// Sample 2: [0, 1, 0, 0]
X[1, 1] = 1.0

// Sample 3: [0, 0, 1, 0]
X[2, 2] = 1.0

// Sample 4: [0, 0, 0, 1]
X[3, 3] = 1.0

// ============================================
// Layer 1: Input -> Hidden
// W1[input, hidden], b1[hidden]
// ============================================
W1[i, h] = 0

// Initialize with small random-ish values
W1[0, 0] = 0.1
W1[0, 1] = 0.2
W1[1, 2] = 0.1
W1[1, 3] = 0.2
W1[2, 4] = 0.1
W1[2, 5] = 0.2
W1[3, 6] = 0.1
W1[3, 7] = 0.2

// Hidden pre-activation: Z1[batch, hidden] = X[batch, input] @ W1[input, hidden]
// (bias omitted for simplicity - would need broadcasting support)
Z1[b, h] = X[b, i] W1[i, h]

// Hidden activation: ReLU
H1[b, h] = relu(Z1[b, h])

// ============================================
// Layer 2: Hidden -> Output
// W2[hidden, output], b2[output]
// ============================================
W2[h, o] = 0

// Initialize output weights
W2[0, 0] = 0.3
W2[1, 0] = 0.2
W2[2, 1] = 0.3
W2[3, 1] = 0.2
W2[4, 0] = 0.1
W2[5, 1] = 0.1
W2[6, 0] = 0.2
W2[7, 1] = 0.2

// Output pre-activation: Z2[batch, output] = H1[batch, hidden] @ W2[hidden, output]
// (bias omitted for simplicity)
Z2[b, o] = H1[b, h] W2[h, o]

// Output activation: Softmax (for classification)
Y[b, o] = softmax(Z2[b, o])

// ============================================
// Target labels (one-hot encoded)
// ============================================
T[b, o] = 0

// Sample 1 -> class 0
T[0, 0] = 1.0

// Sample 2 -> class 0
T[1, 0] = 1.0

// Sample 3 -> class 1
T[2, 1] = 1.0

// Sample 4 -> class 1
T[3, 1] = 1.0

// ============================================
// Cross-entropy loss (per sample)
// Loss = -sum(T * log(Y))
// ============================================
// Note: Full loss computation would require log() and sum reduction
// For now, we just have the predictions Y ready for gradient computation
