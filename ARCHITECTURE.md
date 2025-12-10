# Tensor Logic Language - Architecture

## Goal

Create a **standalone compiled language** for tensor logic - not a library, not an interpreter, but a real compiled language that produces native executables via LLVM.

```
program.tl  →  tlc compile  →  program (native binary)
./program                   →  runs without tlc
```

## Architecture: LLVM Only

```
.tl source
    ↓
[Frontend: Lexer → Parser → AST → Type Check]
    ↓
[Codegen: AST → LLVM IR]
    ↓
[LLVM: llc/clang]
    ↓
Native executable
```

**No C backend. No interpreter. LLVM IR directly.**

## Current Status

```
src/
├── frontend/          ✅ COMPLETE
│   ├── lexer.zig      # Tokenization
│   ├── parser.zig     # Parsing to AST
│   ├── ast.zig        # AST definitions
│   ├── tokens.zig     # Token types
│   ├── types.zig      # Type checking
│   └── optimize.zig   # AST optimization
│
├── lsp/               ✅ COMPLETE
│   └── server.zig     # VS Code language server
│
├── codegen/           ✅ COMPLETE (modular)
│   ├── llvm.zig       # Main orchestrator + gradient codegen
│   ├── autodiff.zig   # Computation graph, gradient derivation
│   ├── types.zig      # Shared types (TensorInfo, IndexVar)
│   ├── tensor.zig     # Tensor allocation and indexing
│   ├── expr.zig       # Expression evaluation
│   ├── einsum.zig     # Einstein summation loops
│   ├── softmax.zig    # Softmax with reduction
│   ├── layernorm.zig  # Layer normalization
│   └── fixpoint.zig   # Recursive equation handling
│
└── runtime/           ⚠️  SCAFFOLDING (not the goal)
    └── *.zig          # Interpreter - for testing only, not production
```

## What's Implemented

### LLVM Code Generation (`src/codegen/llvm.zig`)

1. **Tensor Operations** ✅
   - Allocation via calloc (zero-initialized)
   - Multi-dimensional indexing with strides
   - Constant index support

2. **Einstein Summation (Core)** ✅
   - Index analysis: free indices vs contracted indices
   - Nested loop generation for contractions
   - Accumulation operators: `=`, `+=`, `max=`, `min=`, `*=`

3. **Nonlinearities** ✅
   - All basic: step, relu, sigmoid, tanh, exp, log, sqrt, abs, sin, cos
   - Softmax with reduction across normalization axis
   - Layer normalization with reduction across normalization axis

4. **Control Flow** ✅
   - Fixpoint iteration for recursive rules
   - Automatic detection of recursive equations
   - Convergence detection with change tracking

5. **Autodiff** ✅
   - Reverse-mode automatic differentiation
   - `backward L wrt W, X` syntax
   - Computation graph analysis
   - Chain rule gradient propagation
   - Gradient rules: relu, sigmoid, matmul, pass-through

6. **Future Work**
   - Sparse tensor support in codegen
   - Full matmul gradient (A*B not just Y*Y)

## CLI Design

```bash
# Compile to executable (primary use case)
tlc compile program.tl -o program

# Emit LLVM IR (for debugging/inspection)
tlc compile program.tl --emit-llvm -o program.ll

# Type check only
tlc check program.tl
```

## Key Principles

1. **Standalone executables** - Output runs without tlc installed
2. **No runtime dependency** - All tensor ops compile to native code
3. **LLVM native** - Leverage LLVM's optimizations directly
4. **Performance first** - This is a compiled language, not a scripting language

## Non-Goals

- ~~C code generation~~ (LLVM only)
- ~~Interpreter as production feature~~ (scaffolding only)
- JIT compilation
- Garbage collection (tensors are explicitly managed)
- Dynamic typing (all shapes known at compile time)

## Success Criteria

The project is complete when:

1. `tlc compile hello.tl -o hello` produces a working native binary via LLVM
2. `./hello` runs without tlc, zig, or any interpreter
3. All tensor operations execute as native code
4. Einstein summation compiles to efficient nested loops
5. Performance is competitive with hand-written C
