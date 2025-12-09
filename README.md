# Tensor Logic Compiler (tlc)

A compiler and runtime for Tensor Logic - a unified language for AI that combines neural networks, logic, and probabilistic inference through tensor operations.

Based on Pedro Domingos' paper "[Tensor Logic: The Language of AI](https://arxiv.org/abs/2307.01567)".

## Overview

Tensor Logic represents relations as tensors and uses Einstein summation notation for expressing computations. This enables:

- **Neural Networks**: Express MLPs, attention, GNNs as tensor operations
- **Logic Programming**: Relations as Boolean tensors, inference as matrix operations
- **Probabilistic Reasoning**: Weighted model counting, belief propagation
- **Automatic Differentiation**: Gradient-based learning on all operations

## Quick Start

```bash
# Build the compiler
zig build

# Run a program
./zig-out/bin/tlc run -f examples/matmul.tl

# Start the REPL
./zig-out/bin/tlc repl

# Run with fixpoint iteration (for recursive relations)
./zig-out/bin/tlc run -f examples/ancestor.tl -p
```

## Language Syntax

### Domain Declarations

Define the size of tensor indices:

```
domain i: 10    // Index i ranges from 0 to 9
domain j: 5
```

### Tensor Initialization

```
A[i,j] = 0      // Initialize 10x5 tensor to zeros
B[i] = 1.5      // Initialize 10-element vector to 1.5
```

### Einstein Summation

Repeated indices are summed over (contracted):

```
// Matrix multiplication: C[i,k] = sum_j(A[i,j] * B[j,k])
C[i,k] = A[i,j] B[j,k]

// Dot product: s = sum_i(a[i] * b[i])
s = a[i] b[i]

// Outer product: C[i,j] = a[i] * b[j]
C[i,j] = a[i] b[j]
```

### Nonlinearities

```
Y[i] = relu(X[i])
Y[i] = sigmoid(X[i])
Y[i] = softmax(X[i])
Y[i] = tanh(X[i])
Y[i] = exp(X[i])
Y[i] = log(X[i])
Y[i] = abs(X[i])
Y[i] = sqrt(X[i])
```

### Accumulation Operators

```
A[i,k] += B[i,j] C[j,k]    // Sum accumulation
A[i,k] max= B[i,j] C[j,k]  // Max accumulation
A[i,k] min= B[i,j] C[j,k]  // Min accumulation
```

### Conditionals

```
Y[i] = if X[i] > 0 then X[i] else 0  // Element-wise conditional
```

### Sparse Tensors

```
sparse Parent[i,j]         // Declare sparse relation
Parent[0,1] = 1           // Set specific entries
Parent[2,3] = 1
```

## Examples

### Matrix Multiplication

```
domain i: 3
domain j: 4
domain k: 5

A[i,j] = 0
B[j,k] = 0

// Fill matrices...
A[0,0] = 1
B[0,0] = 2

// Matrix multiply
C[i,k] = A[i,j] B[j,k]
```

### Neural Network Layer

```
domain batch: 32
domain in: 784
domain hidden: 256
domain out: 10

// Forward pass
H[batch,hidden] = relu(X[batch,in] W1[in,hidden] + B1[hidden])
Y[batch,out] = softmax(H[batch,hidden] W2[hidden,out] + B2[out])
```

### Transitive Closure (Ancestor Relation)

```
domain person: 5

sparse Parent[i,j]
Parent[0,1] = 1    // 0 is parent of 1
Parent[1,2] = 1    // 1 is parent of 2

// Ancestor = transitive closure of Parent
Ancestor[i,j] = Parent[i,j]
Ancestor[i,k] max= Ancestor[i,j] Parent[j,k]  // Fixpoint iteration
```

### Self-Attention

```
domain seq: 128
domain dim: 64

// Attention scores
Attn[q,k] = softmax(Q[q,dim] K[k,dim] / sqrt(64))

// Attended values
Out[q,dim] = Attn[q,k] V[k,dim]
```

## CLI Options

```
tlc run -f <file>     Run a .tl file
tlc run -f <file> -p  Run with fixpoint iteration
tlc run -f <file> -t  Run with trace output
tlc run -f <file> -O  Run with AST optimization
tlc repl              Start interactive REPL
tlc help              Show help
```

### REPL Commands

```
:help        Show help
:show        Show all tensor values
:show X      Show tensor X
:clear       Clear all tensors
:fixpoint    Run fixpoint iteration
:load file   Load a .tl file
:quit        Exit REPL
```

## Architecture

```
src/
├── main.zig              # CLI entry point
├── frontend/
│   ├── lexer.zig         # Tokenizer
│   ├── parser.zig        # Parser
│   ├── ast.zig           # AST definitions
│   ├── tokens.zig        # Token types
│   ├── types.zig         # Type checker
│   └── optimize.zig      # AST optimizer
└── runtime/
    ├── interpreter.zig   # Evaluator
    ├── tensor.zig        # Tensor storage
    ├── einsum.zig        # Einstein summation
    ├── autodiff.zig      # Automatic differentiation
    ├── optimizer.zig     # SGD, Adam optimizers
    ├── probability.zig   # Probabilistic inference
    └── training.zig      # Training utilities
```

## Features

### Implemented

- [x] Lexer and parser for .tl files
- [x] Dense and sparse tensor storage
- [x] Einstein summation (einsum) operations
- [x] Matrix operations (multiply, transpose, add)
- [x] Nonlinearities (relu, sigmoid, softmax, tanh, exp, log)
- [x] Accumulation operators (+=, max=, min=)
- [x] Fixpoint iteration for recursive relations
- [x] REPL with command history
- [x] Detailed error messages with source locations
- [x] Type checking and shape inference
- [x] Automatic differentiation (reverse-mode AD)
- [x] Optimizers (SGD, Momentum, Adam)
- [x] Broadcasting for tensor operations
- [x] Conditionals (if-then-else)
- [x] AST optimization (constant folding, strength reduction)
- [x] Probabilistic inference (WMC, belief propagation)
- [x] Training utilities

### Examples Included

- `matmul.tl` - Matrix multiplication
- `ancestor.tl` - Transitive closure
- `mlp.tl` - Multi-layer perceptron
- `attention.tl` - Self-attention mechanism
- `transformer.tl` - Transformer block
- `gnn.tl` - Graph neural network
- `neural_net.tl` - Complete neural network example
- `broadcasting.tl` - Broadcasting and conditionals
- `training.tl` - Gradient-based training

## Building

Requires Zig 0.15.0 or later.

```bash
# Build
zig build

# Run tests
zig build test

# Build release
zig build -Doptimize=ReleaseFast
```

## Testing

```bash
# Run all tests
zig build test

# The test suite covers:
# - Lexer/parser
# - Type checking
# - Tensor operations
# - Einsum contractions
# - Autodiff gradients
# - Optimizers
# - Probabilistic inference
```

## References

- Domingos, P. (2023). [Tensor Logic: The Language of AI](https://arxiv.org/abs/2307.01567)
- Einstein summation notation
- Automatic differentiation (reverse-mode/backpropagation)
- Belief propagation and factor graphs

## License

MIT
