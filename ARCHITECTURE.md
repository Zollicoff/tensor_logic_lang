# Tensor Logic Language - Architecture

> Based on Pedro Domingos' paper "Tensor Logic: The Language of AI"

## Core Philosophy

> "The sole construct in tensor logic is the tensor equation"

This is a **standalone compiled language** - not a library, not an interpreter. It produces native executables via LLVM.

```
program.tl  →  tlc compile  →  LLVM IR  →  clang  →  native binary
```

## Two Inference Modes (Both Compiled)

The paper describes two inference strategies. Both compile to native code:

### Forward Chaining (Implemented)
- Compiles to **nested loops**
- Computes ALL tensor elements eagerly
- Iterates until fixpoint for recursive equations
- Optimal for: neural networks, dense tensors

### Backward Chaining (Planned)
- Compiles to **recursive functions**
- Computes ONLY what's queried (demand-driven)
- Memoization to avoid redundant computation
- Optimal for: logic queries, sparse knowledge bases

```
# Forward: compute everything
Ancestor?           # → loops over all i,j

# Backward: query-driven
Ancestor[0,5]?      # → recursive call, traces dependencies
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
├── frontend/             # Parsing and analysis
│   ├── lexer.zig         # Tokenization
│   ├── parser.zig        # Recursive descent parser
│   ├── ast.zig           # AST node definitions
│   ├── tokens.zig        # Token types
│   ├── types.zig         # Type checking and inference
│   └── optimize.zig      # AST optimization passes
│
├── codegen/              # LLVM IR generation
│   ├── llvm.zig          # Main orchestrator
│   ├── autodiff.zig      # Computation graph, gradient derivation
│   ├── einsum.zig        # Einstein summation (forward: loops)
│   ├── backward.zig      # Backward chaining (recursive fns + memoization)
│   ├── softmax.zig       # Softmax with reduction
│   ├── layernorm.zig     # Layer normalization
│   ├── concat.zig        # Concatenation for attention heads
│   ├── fixpoint.zig      # Recursive equation convergence
│   ├── sparse.zig        # Sparse tensor support
│   ├── tucker.zig        # Tucker decomposition for sparse→dense scaling
│   ├── bp.zig            # Belief propagation helpers
│   ├── gpu.zig           # GPU backends (CUDA/Metal templates)
│   ├── tensor.zig        # Tensor allocation and indexing
│   ├── expr.zig          # Expression evaluation
│   └── types.zig         # Shared types
│
└── lsp/                  # IDE support
    └── server.zig        # VS Code language server
```

## Implementation Status

### Complete ✅
- Lexer, parser, AST, type checker
- Einstein summation with implicit contraction
- All nonlinearities: step, relu, sigmoid, tanh, softmax, lnorm, exp, log, sqrt, abs, sin, cos
- Accumulation operators: `=`, `+=`, `max=`, `min=`, `*=`, `avg=`
- Division indices `X/2` for pooling
- Slice indices `X[4:8]` for subranges
- Concat for attention head merging
- Forward chaining with fixpoint iteration
- **Backward chaining with memoization** (recursive functions for query-driven inference)
- **Full autodiff**: tanh, exp, log, softmax gradients
- **Temperature sigmoid**: `sigmoid(x, T)` for embedding space reasoning
- **Sparse tensor support** (COO format allocation)
- **File I/O**: `save`/`load` for tensor persistence
- Virtual indices `*t`, primed indices `p'`, index arithmetic `i+1`
- VS Code extension with LSP
- **Tucker decomposition**: `tucker T(r1, r2, r3) from Source` for sparse→dense scaling
- **Belief propagation**: Loopy BP is forward chaining (fixpoint + bp.zig helpers)

### In Progress 🔧
- GPU backends (CUDA/Metal) - kernel templates created, full runtime pending

## Paper Features Mapping

| Paper Feature | Status | Implementation |
|--------------|--------|----------------|
| Tensor equations | ✅ | Core syntax |
| Einstein summation | ✅ | einsum.zig |
| Forward chaining | ✅ | loops + fixpoint |
| Backward chaining | ✅ | recursive functions + memoization |
| Autodiff | ✅ | autodiff.zig (full) |
| Sparse tensors | ✅ | sparse.zig (COO format) |
| Temperature σ(x,T) | ✅ | sigmoid(x, T) for embedding reasoning |
| Tucker decomposition | ✅ | tucker.zig (core tensor + factor matrices) |
| Belief propagation | ✅ | fixpoint.zig + bp.zig (loopy BP = forward chaining) |
| GPU acceleration | 🔧 | gpu.zig (CUDA/Metal templates) |

## CLI

```bash
tlc build program.tl -o program   # Compile to native binary
tlc compile program.tl -o out.ll  # Compile to LLVM IR
tlc check program.tl              # Type check only
tlc lex program.tl                # Show tokens
tlc parse program.tl              # Show AST
```

## Design Principles

1. **One construct**: Everything is a tensor equation
2. **Compiled**: Native binaries via LLVM, no interpreter
3. **Two modes**: Forward (loops) and backward (recursion) chaining
4. **Declarative**: Equations state what, not how
5. **Differentiable**: Gradients are also tensor equations
6. **Faithful to paper**: 100% implementation of Domingos' Tensor Logic
