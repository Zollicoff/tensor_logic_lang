# Tensor Logic Language - Architecture

## Core Philosophy

> "The sole construct in tensor logic is the tensor equation"

This is a **standalone compiled language** - not a library, not an interpreter. It produces native executables via LLVM.

```
program.tl  →  tlc compile  →  LLVM IR  →  clang  →  native binary
```

## Pipeline

```
.tl source
    ↓
[Frontend: Lexer → Parser → AST → Type Check]
    ↓
[Codegen: AST → LLVM IR]
    ↓
[LLVM: clang]
    ↓
Native executable
```

## Source Structure

```
src/
├── main.zig              # CLI entry point (~400 lines)
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
│   ├── llvm.zig          # Main orchestrator + gradient codegen
│   ├── autodiff.zig      # Computation graph, gradient derivation
│   ├── types.zig         # Shared types (TensorInfo, IndexVar)
│   ├── tensor.zig        # Tensor allocation and indexing
│   ├── expr.zig          # Expression evaluation
│   ├── einsum.zig        # Einstein summation loops
│   ├── softmax.zig       # Softmax with reduction
│   ├── layernorm.zig     # Layer normalization
│   └── fixpoint.zig      # Recursive equation handling
│
└── lsp/                  # IDE support
    └── server.zig        # VS Code language server
```

## What's Implemented

### Frontend
- Complete lexer and parser for .tl syntax
- Type checking with shape inference
- AST optimization (constant folding)

### Codegen
- **Tensor Operations**: Allocation, multi-dimensional indexing
- **Einstein Summation**: Free vs contracted index analysis, nested loop generation
- **Accumulation**: `=`, `+=`, `max=`, `min=`, `*=`
- **Nonlinearities**: step, relu, sigmoid, tanh, exp, log, sqrt, abs, sin, cos
- **Reductions**: softmax, layer normalization (with normalization axis)
- **Fixpoint**: Automatic detection of recursive equations, convergence loop
- **Autodiff**: Reverse-mode differentiation via `backward L wrt X, W`

### LSP
- Syntax highlighting
- Real-time error diagnostics
- Hover information

## CLI

```bash
tlc compile program.tl -o program.ll   # Compile to LLVM IR
tlc check program.tl                   # Type check only
tlc lex program.tl                     # Show tokens
tlc parse program.tl                   # Show AST
```

## Design Principles

1. **One construct**: Everything is a tensor equation
2. **Compiled**: Native binaries, no runtime interpreter
3. **Minimal**: No special constructs for loops, conditionals, etc.
4. **Declarative**: Equations state what, not how
5. **Differentiable**: Gradients are also tensor equations

## Non-Goals

- Interpreter (removed)
- REPL (removed)
- JIT compilation
- Garbage collection
- Dynamic typing
