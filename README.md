# Tensor Logic

[![CI](https://github.com/Zollicoff/tensor_logic_lang/actions/workflows/ci.yml/badge.svg)](https://github.com/Zollicoff/tensor_logic_lang/actions/workflows/ci.yml)

A compiled language where everything is a tensor equation.

Based on Pedro Domingos' paper "[Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269)".

> "The sole construct in tensor logic is the tensor equation"

## What is Tensor Logic?

Tensor Logic unifies neural networks and logic programming through a single construct: **the tensor equation**. Relations are tensors, inference is matrix multiplication, and learning is automatic differentiation.

```
// Neural network layer
Hidden[b,h] = relu(Input[b,i] W[i,h])

// Transitive closure (logic programming)
Ancestor[x,z] max= step(Ancestor[x,y] Parent[y,z])

// Self-attention
Attn[q,k.] = softmax(Q[q,d] K[k,d] / 8.0)
```

## Installation

### Prebuilt Binaries

Download the latest release for your platform from [Releases](https://github.com/Zollicoff/tensor_logic_lang/releases):
- `tlc-linux-x86_64` - Linux
- `tlc-macos-x86_64` - macOS Intel
- `tlc-macos-aarch64` - macOS Apple Silicon

### Build from Source

```bash
# Requires Zig 0.15+ and Clang
zig build
```

## Quick Start

```bash
# Compile and run
./zig-out/bin/tlc build examples/matmul.tl -o matmul
./matmul
```

## Language Reference

### Domains

Define index ranges:

```
domain b: 32
domain i: 784
domain h: 256
domain o: 10
```

### Tensor Equations

The only statement type. Repeated indices are summed (Einstein notation):

```
// Matrix multiply: C[i,k] = Σⱼ A[i,j] * B[j,k]
C[i,k] = A[i,j] B[j,k]

// Dot product: s = Σᵢ a[i] * b[i]
s = a[i] b[i]

// Outer product
C[i,j] = a[i] b[j]
```

### Accumulation Operators

```
A[i,k] = B[i,j] C[j,k]       // Assignment
A[i,k] += B[i,j] C[j,k]      // Sum (default for repeated indices)
A[i,k] *= B[i,j] C[j,k]      // Product
A[i,k] max= B[i,j] C[j,k]    // Max (for fixpoint iteration)
A[i,k] min= B[i,j] C[j,k]    // Min
A[i,k] avg= B[i,j] C[j,k]    // Average
```

### Nonlinearities

```
Y[i] = relu(X[i])
Y[i] = sigmoid(X[i])
Y[i] = sigmoid(X[i], 0.5)    // Temperature sigmoid σ(x/T)
Y[i] = tanh(X[i])
Y[i] = step(X[i])            // Heaviside (Boolean logic)
Y[i] = exp(X[i])
Y[i] = log(X[i])
Y[i] = abs(X[i])
Y[i] = sqrt(X[i])
Y[i,j.] = softmax(X[i,j])    // Normalize over j
Y[i,j.] = lnorm(X[i,j])      // Layer normalization
```

### Index Types

```
T[i]          // Named index
T[0]          // Constant index
T[i+1]        // Arithmetic (shift)
T[i']         // Primed (independent loop)
T[*t]         // Virtual (embedding lookup)
T[i/2]        // Division
T[0:10]       // Slice
```

### Sparse Tensors

```
sparse Parent[Person, Person]
Parent[0, 1] = 1.0
Parent[1, 2] = 1.0
```

### Automatic Differentiation

```
// Forward pass
H[b,h] = relu(X[b,i] W1[i,h])
Y[b,o] = softmax(H[b,h] W2[h,o])
L = (Y[b,o] - Target[b,o]) * (Y[b,o] - Target[b,o])

// Backward pass - computes dL/dW1, dL/dW2
backward L wrt W1, W2
```

### Backward Chaining

Query-driven recursive inference with memoization:

```
// Define recursive relation
Ancestor[x,y] = Parent[x,y]
Ancestor[x,z] max= Ancestor[x,y] Parent[y,z]

// Query specific values (computes only what's needed)
// Constant indices trigger backward chaining automatically
Ancestor[0, 5]?
```

### File I/O

```
save W "weights.bin"
load W "weights.bin"
```

## Examples

### Neural Network

```
domain b: 32
domain i: 784
domain h: 256
domain o: 10

H[b,h] = relu(X[b,i] W1[i,h])
Y[b,o] = softmax(H[b,h] W2[h,o])
```

### Transformer Attention

```
domain s: 128
domain d: 64

// Self-attention
Scores[q,k] = Q[q,d] K[k,d]
Attn[q,k] = softmax(Scores[q,k])
Out[q,d] = Attn[q,k] V[k,d]
```

### Logic Programming

```
domain n: 100

sparse Edge[n, n]
Edge[0,1] = 1
Edge[1,2] = 1
Edge[2,3] = 1

// Reachability (transitive closure)
Reach[x,y] = step(Edge[x,y])
Reach[x,z] max= step(Reach[x,y] Edge[y,z])
```

### Probabilistic Inference

```
domain s: 10

// Belief propagation is forward chaining
Belief[s] = Prior[s]
Belief[t] max= Belief[s] Transition[s,t] Emission[t,obs]
```

## CLI

```
tlc build <file> -o <output>   Compile to native binary
tlc compile <file> -o <out.ll> Compile to LLVM IR
tlc check <file>               Type check
tlc parse <file>               Show AST
tlc lex <file>                 Show tokens
```

## VS Code Extension

Install the extension from `editors/vscode/` for:
- Syntax highlighting
- Hover documentation
- Go-to-definition
- Autocomplete
- Real-time error diagnostics

## Development

```bash
zig build                        # Debug build
zig build -Doptimize=ReleaseFast # Release build
zig build test                   # Run tests
```

## License

MIT
