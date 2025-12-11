// File I/O - Save and Load Tensors
//
// Demonstrates saving trained weights and loading them back.
// Useful for:
// - Checkpointing during training
// - Deploying trained models
// - Sharing learned parameters

domain i: 4
domain h: 3
domain o: 2

// Initialize weights with some values
W1[i,h] = 0.5
W1[0,0] = 0.1
W1[1,1] = 0.2
W1[2,2] = 0.3
W1[3,0] = 0.4

W2[h,o] = 0.5
W2[0,0] = 0.6
W2[1,1] = 0.7
W2[2,0] = 0.8

// Biases
b1[h] = 0.1
b2[o] = 0.2

// Save weights to binary files
save W1 "w1.bin"
save W2 "w2.bin"
save b1 "b1.bin"
save b2 "b2.bin"

// In a real scenario, you would load these in a separate program:
// load W1 "w1.bin"
// load W2 "w2.bin"

// Query to verify values before save
W1?
W2?
