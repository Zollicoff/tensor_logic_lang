# Changelog

## 2025-12-10 - Interpreter Removal (Back to Core Vision)

**Removed: Interpreter, REPL, and all runtime modules**

The original paper states: "The sole construct in tensor logic is the tensor equation."

We had accumulated features that don't belong in the core language:
- Interpreter (`src/runtime/interpreter.zig`)
- REPL mode
- Built-in optimizers (SGD, Adam)
- Training utilities
- Probability inference modules

These have been **deleted**. The language is now purely compiled:
- `tlc compile` is the primary (and only execution) command
- No `run` or `repl` commands
- Optimizers, training loops, etc. are expressible AS tensor equations

**Files removed:**
- `src/runtime/` directory (6,468 lines)
  - interpreter.zig
  - tensor.zig
  - einsum.zig
  - autodiff.zig
  - optimizer.zig
  - probability.zig
  - training.zig

**Codebase now:**
```
src/main.zig        ~400 lines (CLI)
src/frontend/       ~3,700 lines (lexer, parser, types)
src/codegen/        ~3,300 lines (LLVM IR generation)
src/lsp/            ~700 lines (VS Code integration)
Total:              ~8,100 lines (down from ~14,500)
```

**Updated documentation:**
- README.md - Removed interpreter/REPL references
- ARCHITECTURE.md - Clarified compiled-only philosophy
- build.zig - Removed runtime test targets

---

## Current State (2025-12-09)

**LLVM Codegen is COMPLETE and MODULAR with AUTODIFF**

All core features implemented and working:
- ✅ Matrix multiplication (einsum contraction)
- ✅ Softmax with reduction
- ✅ Layer normalization with reduction
- ✅ Fixpoint iteration (transitive closure)
- ✅ All accumulation operators: `=`, `+=`, `max=`, `min=`, `*=`
- ✅ All nonlinearities: relu, sigmoid, tanh, step, exp, log, sqrt, abs, sin, cos
- ✅ **Autodiff (reverse-mode automatic differentiation)**

Modular structure (`src/codegen/`):
```
llvm.zig       ~650 lines  (main orchestrator + gradient codegen)
autodiff.zig   ~350 lines  (computation graph, gradient derivation)
fixpoint.zig   297 lines   (recursive equations)
layernorm.zig  ~300 lines  (layer normalization)
softmax.zig    267 lines   (softmax with reduction)
expr.zig       243 lines   (expression evaluation)
einsum.zig     191 lines   (loop generation)
tensor.zig     125 lines   (allocation/indexing)
types.zig       35 lines   (shared types)
```

Test files in `/tmp/claude/`:
- `test_matmul.tl` - Matrix multiplication
- `test_softmax.tl` - Softmax operation
- `test_layernorm.tl` - Layer normalization
- `test_transitive.tl` - Transitive closure (fixpoint)
- `test_autodiff.tl` - Automatic differentiation

Future work:
- Sparse tensor support in codegen
- Full matmul gradient (A*B not just Y*Y)

---

## 2025-12-09 - Autodiff (Automatic Differentiation)

**Implemented: Reverse-mode autodiff**

New syntax:
```
backward L wrt X, W
```

This derives gradient equations automatically using the chain rule:
- Builds computation graph from forward equations
- Performs reverse topological sort for backpropagation
- Generates gradient tensor equations

Supported gradient rules:
- `matmul_self`: L = Y*Y -> dL/dY = 2*Y
- `relu_grad`: Y = relu(X) -> dL/dX = dL/dY * step(X)
- `sigmoid_grad`: Y = sigmoid(X) -> dL/dX = dL/dY * Y * (1-Y)
- `pass_through`: Y = X -> dL/dX = dL/dY

New files:
- `src/codegen/autodiff.zig` - Computation graph and gradient derivation
- Updated `src/codegen/llvm.zig` - Gradient code generation

Example:
```
X[0] = -1; X[1] = 2; X[2] = 3
Y[i] = relu(X[i])
L = Y[i] Y[i]
backward L wrt X
dL_dX?  // Output: [0, 4, 6]
```

Test verified: dL/dX = 2*Y * step(X) = [0, 4, 6] (correct!)

---

## 2025-12-09 - Layer Normalization

**Implemented: Layer Normalization (`lnorm`)**
- Three-pass algorithm: compute mean, compute variance, normalize
- Formula: `(x - mean) / sqrt(variance + epsilon)` where epsilon = 1e-5
- Uses normalize axis notation (`j.`) same as softmax
- Works with batch dimensions

New files:
- `src/codegen/layernorm.zig` - Layer normalization codegen (~300 lines)

Example usage:
```
Y[i, j.] = lnorm(X[i, j])  // normalizes each row over j
```

Test results:
- `[1, 2, 3]` -> `[-1.225, 0, 1.225]` (correct)
- `[4, 4, 4]` -> `[0, 0, 0]` (uniform input -> all zeros)

---

## 2025-12-09 - LLVM Codegen Implementation

### Major Changes

**Vision Clarified: LLVM Only**
- Updated ARCHITECTURE.md to reflect the correct goal: standalone compiled language via LLVM
- No C backend, no interpreter as production feature
- Goal: `tlc compile program.tl -o program` produces native binary via LLVM

**LLVM Code Generator (`src/codegen/llvm.zig`)**
- Complete rewrite of LLVM IR emission (~750 lines)
- Core einsum loop generation working:
  - Index analysis: identifies free vs contracted indices
  - Nested loop generation for all indices
  - Accumulation for contracted indices (implicit summation)
- All accumulation operators: `=`, `+=`, `max=`, `min=`, `*=`
- All nonlinearities: relu, sigmoid, tanh, step, exp, log, sqrt, abs, sin, cos
- Tensor allocation via `calloc` (zero-initialized)
- Query statement support (prints tensor values)

**CLI Updates (`src/main.zig`)**
- Added `tlc compile <file>` command (the primary command)
- Added `-o <output>` flag for output file
- Updated help text to prioritize compilation over interpretation
- Interpreter commands moved to "for testing only" status

**Bug Fixes**
- Fixed memory leak in codegen using arena allocator for temp strings
- Updated all code to Zig 0.15 API (ArrayListUnmanaged, StringHashMapUnmanaged)

### Architecture

```
.tl source
    ↓
[Frontend: Lexer → Parser → AST → Type Check]  ✅ Complete
    ↓
[Codegen: AST → LLVM IR]                        ✅ Working
    ↓
[LLVM: clang]                                   External
    ↓
Native executable
```

### Example Usage

```bash
# Compile to LLVM IR
./zig-out/bin/tlc compile examples/matmul.tl -o matmul.ll

# Build native binary
clang matmul.ll -o matmul -lm

# Run
./matmul
```

### Bug Fixes (2025-12-09 continued)

**Fixed: Constant Index Tensor Dimensions**
- Added pre-pass to scan all equations and compute max dimensions for each tensor
- `A[0,0]=1; A[0,1]=2; A[1,0]=3; A[1,1]=4` now correctly allocates 2x2 tensor

**Fixed: Duplicate String Constants**
- Tensor name strings (`@.name.X`) now emitted only once per tensor
- Prevents LLVM linker errors from duplicate symbols

**Fixed: Scalar Assignment Offsets**
- Constant index assignments now compute correct linear offset
- `A[1,1] = 4` stores at offset 3 (not 0) for row-major 2x2 tensor

**Implemented: Softmax with Reduction**
- Full softmax operation: `Y[i, j.] = softmax(X[i, j])`
- The `.` notation (`j.`) specifies the normalization axis
- Two-pass algorithm: sum exp values, then divide
- Works with any number of batch dimensions

**Fixed: Float Literal Formatting**
- LLVM requires decimal point in float literals
- Now `1.0` emits as `1.0` not `1`

**Implemented: Fixpoint Iteration**
- Recursive equations (LHS tensor appears in RHS) are detected automatically
- Wrapped in fixpoint loop that runs until convergence (or max 1000 iterations)
- Change detection: loop continues only if values actually changed
- For Boolean logic, use `max=` operator (idempotent: max(1,1) = 1)
- Example: transitive closure computes correctly!

### Known Limitations (to be addressed)

1. ~~Constant index assignments (`A[0,1] = 2`) don't expand tensor dimensions correctly~~ FIXED
2. ~~Duplicate `@.name.X` strings generated for multiple equations on same tensor~~ FIXED
3. ~~Softmax needs reduction pass (not yet implemented)~~ FIXED
4. ~~Fixpoint iteration not yet in codegen~~ FIXED

**Refactored: Modular Codegen Structure**
- Split monolithic `llvm.zig` (1400 lines) into focused modules:
  - `llvm.zig` (~400 lines) - Main orchestrator
  - `types.zig` - Shared types (TensorInfo, IndexVar, LoopLabels)
  - `tensor.zig` - Tensor allocation, indexing, stride computation
  - `expr.zig` - Expression evaluation (literals, binary ops, products, nonlinearities)
  - `einsum.zig` - Einstein summation loop generation
  - `softmax.zig` - Softmax with reduction
  - `fixpoint.zig` - Recursive equation fixpoint iteration
- Better separation of concerns
- Easier to maintain and extend
- Each module has single responsibility

### Files Changed

- `src/codegen/llvm.zig` - Main orchestrator (refactored)
- `src/codegen/types.zig` - NEW: Shared types
- `src/codegen/tensor.zig` - NEW: Tensor operations
- `src/codegen/expr.zig` - NEW: Expression evaluation
- `src/codegen/einsum.zig` - NEW: Loop generation
- `src/codegen/softmax.zig` - NEW: Softmax operation
- `src/codegen/fixpoint.zig` - NEW: Recursive equations
- `src/main.zig` - Added compile command
- `ARCHITECTURE.md` - Updated to LLVM-only vision, modular structure
