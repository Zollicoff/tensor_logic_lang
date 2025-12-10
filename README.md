# Tensor Logic Compiler (tlc)

A compiled language for Tensor Logic - where everything is a tensor equation.

Based on Pedro Domingos' paper "[Tensor Logic: The Language of AI](https://arxiv.org/abs/2307.01567)".

> "The sole construct in tensor logic is the tensor equation"

## Overview

Tensor Logic unifies neural networks and logic programming through a single construct: the tensor equation. Relations are tensors, inference is matrix multiplication, and everything compiles to native code via LLVM.

```
program.tl  →  tlc compile  →  LLVM IR  →  clang  →  native binary
```

## Quick Start

```bash
# Build the compiler
zig build

# Compile a program
./zig-out/bin/tlc compile examples/matmul.tl -o matmul.ll

# Build native binary
clang matmul.ll -o matmul -lm

# Run
./matmul
```

## Language Syntax

### Domain Declarations

```
domain I(10)    # Index I ranges from 0 to 9
domain J(5)
```

### Tensor Equations

The only statement type:

```
LHS = [nonlinearity(] RHS [)]
```

### Einstein Summation

Repeated indices are contracted (summed):

```
# Matrix multiplication: C[i,k] = sum_j(A[i,j] * B[j,k])
C[I,K] = A[I,J] B[J,K]

# Dot product: s = sum_i(a[i] * b[i])
s = a[I] b[I]

# Outer product
C[I,J] = a[I] b[J]
```

### Nonlinearities

```
Y[I] = relu(X[I])
Y[I] = sigmoid(X[I])
Y[I,J.] = softmax(X[I,J])    # J. = normalization axis
Y[I,J.] = lnorm(X[I,J])      # layer normalization
Y[I] = tanh(X[I])
Y[I] = step(X[I])            # Heaviside step (for Boolean logic)
Y[I] = exp(X[I])
Y[I] = log(X[I])
```

### Accumulation Operators

```
A[I,K] += B[I,J] C[J,K]      # Sum accumulation (default)
A[I,K] max= B[I,J] C[J,K]    # Max accumulation (for fixpoint)
A[I,K] min= B[I,J] C[J,K]    # Min accumulation
A[I,K] *= B[I,J] C[J,K]      # Product accumulation
```

### Automatic Differentiation

```
backward L wrt W, X          # Compute gradients dL/dW, dL/dX
```

## Examples

### Matrix Multiplication

```
domain I(3)
domain J(4)
domain K(5)

tensor A[I,J] = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
tensor B[J,K] = [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0]]

C[I,K] = A[I,J] B[J,K]
```

### Neural Network with Training

```
domain B(32)
domain I(784)
domain H(256)
domain O(10)

# Forward pass
Hidden[B,H] = relu(X[B,I] W1[I,H])
Y[B,O] = softmax(Hidden[B,H] W2[H,O])

# Loss
L = (Y[B,O] - Target[B,O]) (Y[B,O] - Target[B,O])

# Compute gradients
backward L wrt W1, W2
```

### Transitive Closure (Logic Programming)

```
domain P(5)

# Parent relation (sparse)
sparse Parent[P,P]
Parent[0,1] = 1
Parent[1,2] = 1

# Ancestor = transitive closure
Ancestor[I,J] = step(Parent[I,J])
Ancestor[I,K] max= step(Ancestor[I,J] Parent[J,K])
```

### Self-Attention

```
domain S(128)
domain D(64)

Attn[Q,K.] = softmax(Query[Q,D] Key[K,D] / 8.0)
Out[Q,D] = Attn[Q,K] Value[K,D]
```

## CLI Reference

```
tlc compile <file>           Compile to LLVM IR (stdout)
tlc compile <file> -o out.ll Compile to file
tlc check <file>             Type check only
tlc lex <file>               Show tokens
tlc parse <file>             Show AST
tlc help                     Show help
tlc version                  Show version
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
├── codegen/
│   ├── llvm.zig          # Main LLVM IR generator
│   ├── autodiff.zig      # Automatic differentiation
│   ├── einsum.zig        # Einstein summation loops
│   ├── softmax.zig       # Softmax codegen
│   ├── layernorm.zig     # Layer norm codegen
│   ├── fixpoint.zig      # Recursive equation handling
│   ├── tensor.zig        # Tensor allocation
│   ├── expr.zig          # Expression codegen
│   └── types.zig         # Shared types
└── lsp/
    └── server.zig        # VS Code language server
```

## Features

- [x] Einstein summation (implicit contraction)
- [x] All nonlinearities (relu, sigmoid, softmax, tanh, step, exp, log, lnorm)
- [x] Accumulation operators (=, +=, max=, min=, *=)
- [x] Fixpoint iteration for recursive relations
- [x] Reverse-mode automatic differentiation
- [x] Type checking and shape inference
- [x] LLVM IR code generation
- [x] VS Code extension with LSP

## Building

Requires Zig 0.15.0 or later.

```bash
zig build                        # Debug build
zig build -Doptimize=ReleaseFast # Release build
zig build test                   # Run tests
```

## References

- Domingos, P. (2023). [Tensor Logic: The Language of AI](https://arxiv.org/abs/2307.01567)

## License

MIT
