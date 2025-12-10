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
- Concat for attention head merging
- Forward chaining with fixpoint iteration
- **Backward chaining with memoization** (recursive functions for query-driven inference)
- Partial autodiff (2D matmul, relu, sigmoid gradients)
- Virtual indices `*t`, primed indices `p'`, index arithmetic `i+1`
- VS Code extension with LSP

### In Progress 🔧
- **Full autodiff**: Complete gradient rules for all operations
- **Sparse codegen**: Integrate existing sparse.zig (currently falls back to dense)
- **Slice indices**: `X[4:8]` - parser done, codegen needed
- **Temperature sigmoid**: `σ(x, T)` for embedding space reasoning
- **File I/O**: `save`/`load` syntax parsed, codegen needed

### Planned 📋
- Tucker decomposition for scaling sparse→dense
- GPU backends (CUDA/Metal)

## Paper Features Mapping

| Paper Feature | Status | Implementation |
|--------------|--------|----------------|
| Tensor equations | ✅ | Core syntax |
| Einstein summation | ✅ | einsum.zig |
| Forward chaining | ✅ | loops + fixpoint |
| Backward chaining | ✅ | recursive functions + memoization |
| Autodiff | 🔧 | autodiff.zig (partial) |
| Sparse tensors | 🔧 | sparse.zig (not integrated) |
| Temperature σ(x,T) | 📋 | For embedding reasoning |
| Tucker decomposition | 📋 | Scaling strategy |

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
