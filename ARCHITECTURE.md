# Architecture

> "The sole construct in tensor logic is the tensor equation"

## Overview

Tensor Logic is a **compiled language** that produces native executables via LLVM.

```
program.tl  →  tlc compile  →  LLVM IR  →  clang  →  native binary
```

## Two Inference Modes

The compiler supports two inference strategies, both compiled to native code:

### Forward Chaining
- Compiles to **nested loops**
- Computes all tensor elements eagerly
- Iterates until fixpoint for recursive equations
- Optimal for: neural networks, dense tensors

### Backward Chaining
- Compiles to **recursive functions with memoization**
- Computes only what's queried (demand-driven)
- Optimal for: logic queries, sparse knowledge bases

```
// Forward: compute everything
Ancestor?           // loops over all i,j

// Backward: query-driven
Ancestor[0,5]?      // recursive call, traces dependencies
```

## Pipeline

```
.tl source
    ↓
[Frontend: Lexer → Parser → AST → Type Check]
    ↓
[Codegen: AST → LLVM IR]
    │
    ├── Forward mode → nested loops + fixpoint
    └── Backward mode → recursive functions + memoization
    ↓
[LLVM: clang]
    ↓
Native executable
```

## Source Structure

```
src/
├── main.zig              # CLI entry point
│
├── frontend/
│   ├── lexer.zig         # Tokenization
│   ├── parser.zig        # Recursive descent parser
│   ├── ast.zig           # AST node definitions
│   ├── tokens.zig        # Token types
│   ├── types.zig         # Type checking and inference
│   └── optimize.zig      # AST optimization passes
│
├── codegen/
│   ├── llvm.zig          # Main LLVM IR orchestrator
│   ├── einsum.zig        # Einstein summation loops
│   ├── expr.zig          # Expression evaluation
│   ├── tensor.zig        # Tensor allocation and indexing
│   ├── fixpoint.zig      # Recursive equation convergence
│   ├── backward.zig      # Backward chaining (recursive + memoization)
│   ├── autodiff.zig      # Reverse-mode automatic differentiation
│   ├── softmax.zig       # Softmax with numerically stable reduction
│   ├── layernorm.zig     # Layer normalization
│   ├── concat.zig        # Tensor concatenation
│   ├── sparse.zig        # Sparse tensor support (COO format)
│   ├── tucker.zig        # Tucker decomposition
│   ├── bp.zig            # Belief propagation helpers
│   ├── gpu.zig           # GPU backend templates (CUDA/Metal)
│   └── types.zig         # Shared codegen types
│
└── lsp/
    └── server.zig        # Language server for VS Code
```

## Design Principles

1. **One construct**: Everything is a tensor equation
2. **Compiled**: Native binaries via LLVM, no interpreter
3. **Two modes**: Forward (loops) and backward (recursion) chaining
4. **Declarative**: Equations state what, not how
5. **Differentiable**: Gradients are also tensor equations
