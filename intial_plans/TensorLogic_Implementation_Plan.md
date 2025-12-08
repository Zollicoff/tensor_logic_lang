# Tensor Logic: Complete Implementation Plan

> Based on Pedro Domingos' paper "Tensor Logic: The Language of AI" (arXiv:2510.12269v3)

---

## Table of Contents

1. [Vision & Core Thesis](#1-vision--core-thesis)
2. [The One Construct: Tensor Equations](#2-the-one-construct-tensor-equations)
3. [Language Specification](#3-language-specification)
4. [Formal Semantics](#4-formal-semantics)
5. [Type System](#5-type-system)
6. [Inference Engines](#6-inference-engines)
7. [Automatic Differentiation](#7-automatic-differentiation)
8. [Embedding Space Reasoning](#8-embedding-space-reasoning)
9. [Scaling Strategies](#9-scaling-strategies)
10. [Architecture & Implementation](#10-architecture--implementation)
11. [File Format & CLI](#11-file-format--cli)
12. [Example Programs](#12-example-programs)
13. [Implementation Phases](#13-implementation-phases)
14. [Testing Strategy](#14-testing-strategy)

---

## 1. Vision & Core Thesis

### 1.1 What Tensor Logic Is

Tensor Logic is a **new programming language** that unifies neural and symbolic AI at a fundamental level. It is not a library, not a DSL embedded in Python—it is its own thing with its own compiler, runtime, and file format.

It is based on the profound observation that:

1. **A relation is a sparse Boolean tensor**
2. **A Datalog rule is an einsum over Boolean tensors with a step function**
3. **Therefore, logic programming and tensor algebra are fundamentally equivalent**

This means the SAME language can express:
- Neural networks (transformers, CNNs, GNNs, MLPs)
- Symbolic reasoning (Datalog, knowledge bases, planning)
- Kernel machines (SVMs, Gaussian processes)
- Probabilistic graphical models (Bayesian networks, Markov random fields)
- **Novel hybrid models** (reasoning in embedding space)

### 1.2 The Key Insight

From the paper:

> "A Datalog rule is an einsum over Boolean tensors, with a step function applied elementwise to the result."

For example, the rule:
```
Aunt(x, z) ← Sister(x, y), Parent(y, z)
```

Is equivalent to:
```
A[x,z] = H(S[x,y] P[y,z])
```

Where:
- `S[x,y] P[y,z]` is an einsum (join on y, project out y)
- `H()` is the Heaviside step function: H(x) = 1 if x > 0, else 0

### 1.3 Design Principles

1. **Minimalism**: One construct (the tensor equation) to rule them all
2. **Declarative**: Equations state what, not how
3. **Differentiable**: Everything is automatically differentiable
4. **Scalable**: GPU-native, sparse-aware
5. **Transparent**: Reasoning is inspectable at any point
6. **Learnable**: Structure and parameters can be learned from data
7. **Standalone**: A real language with its own toolchain, not a Python library

---

## 2. The One Construct: Tensor Equations

### 2.1 The Only Statement Type

A Tensor Logic program consists entirely of **tensor equations**:

```
LHS = [nonlinearity(] RHS [)]
```

Where:
- **LHS**: A tensor reference with indices
- **RHS**: A series of tensor joins (products)
- **nonlinearity**: Optional elementwise function (step, sig, relu, softmax, etc.)

That's it. There are no keywords, no control flow, no special constructs. Everything reduces to tensor equations.

### 2.2 Implicit Operations

The magic is in what's **implicit**:

| Operation | How It's Expressed |
|-----------|-------------------|
| **Join** (×) | Adjacent tensors: `A[i,j] B[j,k]` joins on j |
| **Projection** (Σ) | Indices on RHS but not LHS are summed out |
| **Union** (∨) | Multiple equations with same LHS are summed |
| **Default values** | Tensor elements are 0 by default |

### 2.3 Notation Conventions

| Notation | Meaning |
|----------|---------|
| `T[i, j, k]` | Numeric tensor (real/int) with indices i, j, k |
| `R(x, y)` | Boolean tensor (relation) - parentheses imply Boolean |
| `T[i, *t]` | Virtual index t (no memory allocated, overwritten) |
| `T[i.]` | Index i is the normalization axis (for softmax) |
| `T[i+1]` | Index arithmetic |

---

## 3. Language Specification

### 3.1 Formal Grammar

```ebnf
(* A program is a sequence of statements *)
Program         = { Statement } ;

(* Statements are tensor equations, queries, or pragmas *)
Statement       = TensorEquation
                | Query
                | TypeDecl
                | FileIO
                | Comment ;

(* THE core construct *)
TensorEquation  = TensorRef "=" Expression ;

(* Query a tensor's value *)
Query           = TensorRef "?" ;

(* Optional type/shape declarations (can be inferred) *)
TypeDecl        = TensorRef ":" Type Shape ;

(* File I/O *)
FileIO          = TensorRef "=" FilePath
                | FilePath "=" TensorRef ;

(* Comments *)
Comment         = "#" { any character } newline ;

(* Tensor references *)
TensorRef       = Identifier "[" IndexList "]"      (* numeric tensor *)
                | Identifier "(" IndexList ")"      (* Boolean tensor/relation *)
                | Identifier ;                       (* scalar *)

IndexList       = [ Index { "," Index } ] ;

Index           = Identifier                         (* variable index *)
                | Integer                            (* constant index *)
                | Identifier IndexOp Integer         (* arithmetic: i+1, i-1 *)
                | "*" Identifier                     (* virtual index *)
                | Identifier "."                     (* normalization axis *)
                | Identifier "'"                     (* primed index (distinct from unprimed) *)
                | Integer ":" Integer                (* slice *)
                | Integer "/" Integer ;              (* integer division for pooling *)

IndexOp         = "+" | "-" ;

(* Expressions *)
Expression      = [ Nonlinearity "(" ] JoinExpr [ ")" ]
                | "(" Expression ")" ;

JoinExpr        = Term { Term }                      (* implicit join/multiply *)
                | JoinExpr "+" JoinExpr              (* addition *)
                | JoinExpr "-" JoinExpr              (* subtraction *)
                | JoinExpr "/" Term ;                (* division *)

Term            = TensorRef
                | Literal
                | "(" JoinExpr ")"
                | "-" Term ;                         (* negation *)

(* Literals *)
Literal         = Number
                | "[" NumberList "]"                 (* vector literal *)
                | "[" "[" NumberList "]" { "," "[" NumberList "]" } "]" ;  (* matrix literal *)

NumberList      = Number { "," Number } ;

Number          = Integer | Float ;

(* Nonlinearities *)
Nonlinearity    = "step" | "H"                       (* Heaviside step function *)
                | "sig" | "sigmoid"                  (* sigmoid *)
                | "relu"                             (* ReLU *)
                | "softmax"                          (* softmax *)
                | "lnorm"                            (* layer norm *)
                | "tanh"                             (* hyperbolic tangent *)
                | "exp"                              (* exponential *)
                | "log"                              (* natural log *)
                | "sqrt"                             (* square root *)
                | "sin" | "cos"                      (* trigonometric *)
                | "concat"                           (* concatenation *)
                | Identifier ;                       (* user-defined *)

(* Types *)
Type            = "real" | "int" | "bool" | "complex" ;

Shape           = "[" DimList "]" ;
DimList         = Dimension { "," Dimension } ;
Dimension       = Integer | Identifier ;             (* concrete or symbolic *)

(* Lexical elements *)
Identifier      = Letter { Letter | Digit | "_" } ;
Integer         = [ "-" ] Digit { Digit } ;
Float           = Integer "." { Digit } [ Exponent ] ;
Exponent        = ( "e" | "E" ) [ "+" | "-" ] Digit { Digit } ;
FilePath        = '"' { any character except '"' } '"' ;
```

### 3.2 Alternative Projection Operators

The default `=` sums over projected indices. Alternatives:

| Operator | Meaning |
|----------|---------|
| `=` or `+=` | Sum projection (default) |
| `max=` | Max projection |
| `min=` | Min projection |
| `avg=` | Average projection |
| `*=` | Product projection |

Example:
```
MaxPool[x/2, y/2] max= Features[x, y]
```

### 3.3 Syntactic Sugar

These are conveniences that compile to basic tensor equations:

| Sugar | Expansion |
|-------|-----------|
| `A + B` | Two equations with same LHS |
| `A - B` | `A + (-1 * B)` |
| `A / B` | Elementwise division (procedural attachment) |
| `A^n` | `A A A...` (n times, or procedural) |
| `Even(d)`, `Odd(d)` | Boolean tensors for conditionals |

---

## 4. Formal Semantics

### 4.1 Tensor Join

The **join** of tensors U and V along common indices β:

```
(U ⋈ V)[α,β,γ] = U[α,β] × V[β,γ]
```

Where:
- α = indices only in U
- β = indices in both U and V (joined on)
- γ = indices only in V

**Special cases:**
- No common indices → Kronecker product (tensor product)
- All indices common → Hadamard product (elementwise)
- Boolean tensors → database join

**In syntax:** Adjacent tensors are joined:
```
A[i,j] B[j,k]  # joins on j, result has indices i,k (and j before projection)
```

### 4.2 Tensor Projection

The **projection** of tensor T onto indices α:

```
π_α(T) = Σ_β T[α,β]
```

Where β are all indices of T not in α.

**In syntax:** Indices on RHS but not on LHS are projected (summed) out:
```
Y[i,k] = A[i,j] B[j,k]   # j is summed out
Y = W[i] X[i]             # i is summed out (dot product)
```

### 4.3 Equation Accumulation

Multiple equations with the same LHS are **summed**:

```
Ancestor(x,z) = step(Parent(x,z))
Ancestor(x,z) = step(Ancestor(x,y) Parent(y,z))
```

Is equivalent to:
```
Ancestor(x,z) = step(Parent(x,z) + Ancestor(x,y) Parent(y,z))
```

For Boolean tensors, this implements **logical OR**.

### 4.4 The Step Function

The **Heaviside step function** H(x):

```
H(x) = 1  if x > 0
     = 0  otherwise
```

This is what makes Boolean tensor operations correspond to logic:
- Join + Step = Logical AND (existential over joined indices)
- Sum + Step = Logical OR

### 4.5 Evaluation Order

**Forward chaining:**
1. Initialize all tensors to defaults (0) or provided values
2. For each equation, compute RHS from current tensor values
3. Update LHS tensor with result (accumulating if multiple equations)
4. Repeat until fixpoint (no changes) or iteration limit

**Backward chaining:**
1. Start with query
2. Find equations that can produce the queried tensor
3. Recursively evaluate RHS tensors
4. Combine results

---

## 5. Type System

### 5.1 Tensor Types

| Type | Description | Default Value |
|------|-------------|---------------|
| `real` | Floating point (f32/f64) | 0.0 |
| `int` | Integer | 0 |
| `bool` | Boolean (0 or 1) | 0 (false) |
| `complex` | Complex number | 0+0i |

### 5.2 Shape System

A tensor's **shape** consists of:
- **Rank**: Number of indices
- **Size**: Number of elements along each index

Shapes can be:
- **Concrete**: `[100, 768]`
- **Symbolic**: `[N, D]` where N, D are defined elsewhere
- **Inferred**: From context

### 5.3 Type Inference Rules

1. **Literals**: Inferred from form (`1` → int, `1.0` → real, `[0,1,1,0]` → bool if all 0/1)
2. **Parentheses notation**: `R(x,y)` implies Boolean type
3. **Operations**: Join of bools → int (count), then step → bool
4. **Nonlinearities**: May change type (step → bool, sig → real in [0,1])

### 5.4 Index Domains

Indices range over **domains** (finite sets). Domains can be:
- **Implicit**: Inferred from tensor dimensions
- **Named**: For documentation/checking

```
# Optional domain declarations
Person: [1000]
Time: [24]
D: [768]

# Tensors use these domains
Parent(Person, Person): bool
Emb[Person, D]: real
```

---

## 6. Inference Engines

### 6.1 Forward Chaining

**Algorithm (pseudocode):**
```
fn forward_chain(program, data, max_iter) {
    var tensors = initialize(data);
    for (0..max_iter) |_| {
        var changed = false;
        for (topological_sort(program.equations)) |equation| {
            const old_value = tensors.get(equation.lhs);
            const new_value = evaluate(equation.rhs, tensors);
            const result = apply_nonlinearity(equation.nonlin, new_value);
            tensors.set(equation.lhs, accumulate(old_value, result, equation.op));
            if (!equal(tensors.get(equation.lhs), old_value)) {
                changed = true;
            }
        }
        if (!changed) break;  // fixpoint reached
    }
    return tensors;
}
```

**Use cases:**
- Computing deductive closure
- Neural network forward pass (single iteration)
- Reaching fixpoint for recursive rules

### 6.2 Backward Chaining

**Algorithm (pseudocode):**
```
fn backward_chain(program, query, data, memo) {
    if (memo.contains(query)) return memo.get(query);
    if (data.contains(query)) return data.get(query);

    var result = default_value(query.type);
    for (equations_for(query, program)) |equation| {
        var rhs_values = [];
        for (equation.rhs_tensors) |t| {
            rhs_values.append(backward_chain(program, t, data, memo));
        }
        const partial = evaluate_with(equation, rhs_values);
        result = accumulate(result, partial, equation.op);
    }

    memo.put(query, result);
    return result;
}
```

**Use cases:**
- Query-driven evaluation (only compute what's needed)
- Avoiding unnecessary computation in large programs

### 6.3 Hybrid Strategies

- **Semi-naive evaluation**: Only recompute based on changes
- **Magic sets**: Transform program to be more goal-directed
- **Stratification**: Handle negation and aggregation correctly

---

## 7. Automatic Differentiation

### 7.1 Core Insight

From the paper:

> "The gradient of a tensor logic program is also a tensor logic program"

This is because tensor equations have a simple derivative structure.

### 7.2 Derivative Rules

For a tensor equation:
```
Y[...] = T[...] X₁[...] ... Xₙ[...]
```

The derivative of Y with respect to T is:
```
∂Y[...]/∂T[...] = X₁[...] ... Xₙ[...]
```

(The product of all other tensors on the RHS)

### 7.3 Gradient Computation

For loss L with respect to tensor T:

```
∂L/∂T = Σ_E (dL/dY)(dY/dU) Π_{U\T} X
```

Where:
- E = equations where T appears on RHS
- Y = LHS of equation
- U = nonlinearity argument
- X = tensors in U

### 7.4 Backpropagation Through Structure

When program structure varies per example (different equations apply):
- Track which equations fired for each example
- Backpropagate through the actual derivation path
- Special case: Backpropagation through time for RNNs

### 7.5 Learning Specification

To learn a tensor logic program:

```
# Define the model
X[i, j] = sig(W[i, j, k] X[i-1, k])

# Define the loss
Loss = (Y[e] - X[*e, N, j])^2

# Mark learnable parameters (optional, default = all non-data tensors)
learn W

# Training loop (built-in or custom)
train(epochs=100, lr=0.01, optimizer="adam")
```

---

## 8. Embedding Space Reasoning

### 8.1 The Vision

This is the **most important new capability** tensor logic enables:
- Combine deductive reasoning (symbolic) with similarity (neural)
- Control the balance via temperature parameter
- Maintain transparency and reliability

### 8.2 Random Embeddings

Objects can be embedded as random unit vectors:
```
Emb[x, d]: real  # embedding matrix

# Retrieve object A's embedding
E[d] = V[x] Emb[x, d]  # V is one-hot for A

# Set membership (like Bloom filter)
S[d] = V[x] Emb[x, d]           # superposition of set elements
D[A] = S[d] Emb[A, d]           # ~1 if A in set, ~0 otherwise
InSet(A) = step(D[A] - 0.5)     # threshold
```

### 8.3 Relation Embedding

Embed a binary relation R(x,y) as a matrix:
```
EmbR[i, j] = R(x, y) Emb[x, i] Emb[y, j]
```

Query the relation:
```
D[A, B] = EmbR[i, j] Emb[A, i] Emb[B, j]  # ~1 if R(A,B), ~0 otherwise
```

This works because:
```
D[A,B] = R(x,y) (Emb[x,i] Emb[A,i]) (Emb[y,j] Emb[B,j])
       ≈ R(A,B)  # since Emb[x,i]Emb[A,i] ≈ δ(x,A)
```

### 8.4 Rule Embedding

Embed a Datalog rule by embedding its relations:
```
# Original rule: Cons(...) ← Ant₁(...), ..., Antₙ(...)
# Embedded version:
EmbCons[...] = EmbAnt1[...] ... EmbAntn[...]
```

### 8.5 Learned Embeddings & Temperature

When embeddings are learned (not random):
```
Sim[x, x'] = Emb[x, d] Emb[x', d]  # similarity matrix
```

Similar objects "borrow" inferences from each other!

**Temperature control:**
```
Y[...] = σ(X[...], T)  # sigmoid with temperature T

# T = 0: Pure deductive reasoning (Gram matrix → identity)
# T > 0: Analogical reasoning (similar things share inferences)
```

### 8.6 Tucker Decomposition

For scaling, convert sparse tensors to dense via Tucker decomposition:
```
A[i, j, k] = M[i, p] M'[j, q] M''[k, r] C[p, q, r]
```

Where C is a compact core tensor and M, M', M'' are factor matrices.

---

## 9. Scaling Strategies

### 9.1 Approach 1: Separation of Concerns

- **Dense tensors**: GPU (einsum, matmul)
- **Sparse tensors**: Database engine (join, project)
- **Mixed**: Database orchestrates, GPU handles dense subproblems

### 9.2 Approach 2: Tucker Decomposition

Convert everything to dense via decomposition:
1. Embed sparse tensors (relations) as dense embeddings
2. All operations become GPU-native einsum
3. Denoise periodically by thresholding

**Advantages:**
- Unified GPU execution
- Exponentially more efficient than naive sparse→dense
- Combines naturally with learned embeddings

### 9.3 Sparse Tensor Representation

Relations are stored as tuples, not dense matrices:
```
Parent = {(Alice, Bob), (Bob, Charlie), (Alice, David)}
# NOT a 1000x1000 matrix with 3 ones
```

Join operations iterate over tuples, not matrix elements.

### 9.4 Compilation Optimizations

- **Common subexpression elimination**
- **Join ordering** (smallest intermediate results first)
- **Predicate pushdown** (filter early)
- **Materialization decisions** (when to store intermediates)

---

## 10. Architecture & Implementation

### 10.1 Implementation Language: Zig

We implement the Tensor Logic compiler in **Zig** because:

1. **Comptime**: Compile-time execution is perfect for tensor shape checking and einsum index resolution
2. **Clean C interop**: Seamless FFI for CUDA, Metal, Vulkan compute
3. **No build system hell**: `zig build` just works
4. **Memory safety without GC**: Manual memory management with safety checks
5. **Cross-compilation**: Easy to build for any platform
6. **LLVM backend**: Good codegen for the compiler itself

### 10.2 Directory Structure

```
tensorlogic/
├── build.zig                    # Zig build configuration
├── README.md
├── LICENSE
│
├── src/
│   ├── main.zig                 # CLI entry point (tlc)
│   │
│   ├── frontend/
│   │   ├── lexer.zig            # Tokenization
│   │   ├── tokens.zig           # Token types
│   │   ├── parser.zig           # Recursive descent parser
│   │   ├── ast.zig              # AST node definitions
│   │   └── errors.zig           # Error types and formatting
│   │
│   ├── analysis/
│   │   ├── symbols.zig          # Symbol table
│   │   ├── types.zig            # Type inference and checking
│   │   ├── shapes.zig           # Shape inference and checking
│   │   └── dependency.zig       # Dependency graph
│   │
│   ├── ir/
│   │   ├── tensor_ir.zig        # Tensor IR representation
│   │   ├── lowering.zig         # AST → IR lowering
│   │   └── optimize.zig         # IR optimizations
│   │
│   ├── codegen/
│   │   ├── einsum.zig           # Einsum generation
│   │   ├── cuda.zig             # CUDA kernel generation
│   │   ├── metal.zig            # Metal shader generation
│   │   └── cpu.zig              # CPU fallback (SIMD)
│   │
│   ├── runtime/
│   │   ├── tensor.zig           # Runtime tensor representation
│   │   ├── sparse.zig           # Sparse tensor/relation storage
│   │   ├── forward.zig          # Forward chaining engine
│   │   ├── backward.zig         # Backward chaining engine
│   │   └── autodiff.zig         # Automatic differentiation
│   │
│   ├── repl/
│   │   └── repl.zig             # Interactive REPL
│   │
│   └── lib/
│       ├── math.zig             # Math functions (step, sigmoid, etc.)
│       └── io.zig               # File I/O for tensors
│
├── tests/
│   ├── lexer_test.zig
│   ├── parser_test.zig
│   ├── type_test.zig
│   ├── eval_test.zig
│   └── fixtures/                # Test .tl files
│       ├── perceptron.tl
│       ├── ancestor.tl
│       └── ...
│
├── examples/
│   ├── perceptron.tl
│   ├── mlp.tl
│   ├── transformer.tl
│   ├── gnn.tl
│   ├── cnn.tl
│   ├── ancestor.tl
│   ├── path.tl
│   ├── kernel_svm.tl
│   ├── bayesian_net.tl
│   └── embedding_reasoning.tl
│
└── docs/
    ├── language_reference.md
    ├── tutorial.md
    └── paper_examples.md
```

### 10.3 Core Data Structures (Zig)

**Token:**
```zig
pub const TokenType = enum {
    // Literals
    integer,
    float,
    string,
    identifier,

    // Brackets
    lbracket,    // [
    rbracket,    // ]
    lparen,      // (
    rparen,      // )

    // Operators
    equals,      // =
    plus,        // +
    minus,       // -
    star,        // *
    slash,       // /
    caret,       // ^
    question,    // ?
    comma,       // ,
    colon,       // :
    dot,         // .
    prime,       // '

    // Compound
    plus_equals,  // +=
    max_equals,   // max=
    min_equals,   // min=
    avg_equals,   // avg=

    // Special
    newline,
    eof,
    comment,
};

pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    line: u32,
    column: u32,
};
```

**AST Nodes:**
```zig
pub const Index = union(enum) {
    name: []const u8,                    // x, i, j
    constant: i64,                       // 0, 1, 42
    arithmetic: struct {                 // i+1, i-1
        base: []const u8,
        op: enum { add, sub },
        offset: i64,
    },
    virtual: []const u8,                 // *t
    normalize: []const u8,               // i.
    primed: []const u8,                  // x'
    slice: struct { start: i64, end: i64 },  // 4:8
    div: struct { index: []const u8, divisor: i64 },  // x/2
};

pub const TensorRef = struct {
    name: []const u8,
    indices: []Index,
    is_boolean: bool,  // () vs []
};

pub const Expression = union(enum) {
    tensor_ref: TensorRef,
    literal: Literal,
    join: []Expression,      // implicit multiplication
    sum: []Expression,       // addition
    negation: *Expression,
    nonlinearity: struct {
        name: []const u8,
        arg: *Expression,
    },
};

pub const Statement = union(enum) {
    equation: struct {
        lhs: TensorRef,
        rhs: Expression,
        accumulator: Accumulator,
    },
    query: TensorRef,
    type_decl: struct {
        tensor: TensorRef,
        dtype: DataType,
        shape: ?[]Dimension,
    },
};

pub const Program = struct {
    statements: []Statement,
    allocator: std.mem.Allocator,
};
```

**Runtime Tensor:**
```zig
pub const DataType = enum {
    bool,
    int32,
    int64,
    float32,
    float64,
    complex64,
};

pub const Tensor = struct {
    name: []const u8,
    dtype: DataType,
    shape: []usize,
    data: TensorData,
    is_sparse: bool,
    requires_grad: bool,

    pub const TensorData = union(enum) {
        dense: []u8,           // raw bytes, interpreted by dtype
        sparse: SparseTensor,  // COO or CSR format
    };
};

pub const SparseTensor = struct {
    indices: [][]usize,  // list of tuples
    values: []u8,        // corresponding values
    nnz: usize,          // number of non-zeros
};
```

### 10.4 Compilation Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │     │   Tokens    │     │    AST      │
│   (.tl)     │────▶│             │────▶│             │
└─────────────┘     └─────────────┘     └─────────────┘
                         Lexer              Parser
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Optimized  │     │  Tensor IR  │     │  Typed AST  │
│     IR      │◀────│             │◀────│             │
└─────────────┘     └─────────────┘     └─────────────┘
                      Lowering         Type Checking
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                    Code Generation                   │
├─────────────┬─────────────┬─────────────────────────┤
│    CUDA     │    Metal    │         CPU             │
│   Kernels   │   Shaders   │     (SIMD/Scalar)       │
└─────────────┴─────────────┴─────────────────────────┘
       │
       ▼
┌─────────────┐
│   Execute   │
│  (Runtime)  │
└─────────────┘
```

---

## 11. File Format & CLI

### 11.1 File Extension

`.tl` - Tensor Logic source files

### 11.2 CLI Commands

```bash
# Run a program (interpret)
tlc run program.tl

# Run with data file
tlc run program.tl --data data.json

# Compile to executable
tlc build program.tl -o program

# Type check only
tlc check program.tl

# Show parsed AST
tlc parse program.tl --ast

# Show IR
tlc parse program.tl --ir

# Interactive REPL
tlc repl

# Compile to CUDA kernels
tlc build program.tl --target cuda -o program.cu

# Train a model
tlc train program.tl --data train.json --epochs 100 --lr 0.01

# Version
tlc --version
```

### 11.3 REPL Interface

```
$ tlc repl
Tensor Logic v0.1.0
>>> W = [0.2, 1.9, -0.7, 3.0]
>>> X = [0, 1, 1, 0]
>>> Y = step(W[i] X[i])
>>> Y?
1.2
>>> Parent(Alice, Bob)
>>> Parent(Bob, Charlie)
>>> Ancestor(x, y) = step(Parent(x, y))
>>> Ancestor(x, z) = step(Ancestor(x, y) Parent(y, z))
>>> Ancestor?
{(Alice, Bob), (Bob, Charlie), (Alice, Charlie)}
>>> Ancestor(Alice, Charlie)?
1
>>> :quit
```

---

## 12. Example Programs

### 12.1 Single-Layer Perceptron

```tensorlogic
# perceptron.tl
W[i]: real
X[i]: real
Y = step(W[i] X[i])
```

### 12.2 Multilayer Perceptron

```tensorlogic
# mlp.tl
# i = layer index, j = input unit, k = output unit
W[i, j, k]: real   # weights for each layer
X[i, j]: real      # activations at each layer

# Each layer's output is sigmoid of weighted sum of previous layer
X[i, k] = sig(W[i, j, k] X[i-1, j])

# Query final layer
X[N, j]?
```

### 12.3 Recurrent Neural Network

```tensorlogic
# rnn.tl
W[i, j]: real      # recurrent weights
V[i, j]: real      # input weights
U[j, t]: real      # input sequence
X[i, *t]: real     # hidden state (virtual time index)

# RNN update equation
X[i, *t+1] = sig(W[i, j] X[j, *t] + V[i, j] U[j, t])
```

### 12.4 Ancestor (Transitive Closure)

```tensorlogic
# ancestor.tl
Parent(x, y): bool

# Base case: parents are ancestors
Ancestor(x, y) = step(Parent(x, y))

# Recursive case: ancestor's child's ancestors
Ancestor(x, z) = step(Ancestor(x, y) Parent(y, z))

# Query
Ancestor?
```

### 12.5 Graph Neural Network

```tensorlogic
# gnn.tl
Neig(x, y): bool           # adjacency relation
X[n, d]: real              # node features
Emb[n, 0, d] = X[n, d]     # initialize embeddings

# Message passing layers
WP[l, d', d]: real         # MLP weights
WAgg[l, d]: real           # aggregation weights
WSelf[l, d]: real          # self-connection weights

# Per-layer computation
Z[n, l, d'] = relu(WP[l, d', d] Emb[n, l, d])
Agg[n, l, d] = Neig(n, n') Z[n', l, d]
Emb[n, l+1, d] = relu(WAgg[l, d] Agg[n, l, d] + WSelf[l, d] Emb[n, l, d])

# Node classification
WOut[d]: real
Y[n] = sig(WOut[d] Emb[n, L, d])
```

### 12.6 Transformer Attention

```tensorlogic
# transformer.tl
X(p, t): bool              # input text (position, token)
Emb[t, d]: real            # token embeddings

# Embed input
EmbX[p, d] = X(p, t) Emb[t, d]

# Positional encoding
PosEnc[p, d] = Even(d) sin(p / L^(d/De)) + Odd(d) cos(p / L^((d-1)/De))

# Residual stream
Stream[0, p, d] = EmbX[p, d] + PosEnc[p, d]

# Attention (per block b, head h)
WQ[b, h, dk, d]: real
WK[b, h, dk, d]: real
WV[b, h, dv, d]: real

Query[b, h, p, dk] = WQ[b, h, dk, d] Stream[b, p, d]
Key[b, h, p, dk] = WK[b, h, dk, d] Stream[b, p, d]
Val[b, h, p, dv] = WV[b, h, dv, d] Stream[b, p, d]

Comp[b, h, p, p'.] = softmax(Query[b, h, p, dk] Key[b, h, p', dk] / sqrt(Dk))
Attn[b, h, p, dv] = Comp[b, h, p, p'] Val[b, h, p', dv]

# Merge heads and add to stream
WS[b, d, dm]: real
Merge[b, p, dm] = concat(Attn[b, h, p, dv])
Stream[b, p, d.] = lnorm(WS[b, d, dm] Merge[b, p, dm] + Stream[b, p, d])

# MLP
WP[b, p, d]: real
MLP[b, p] = relu(WP[b, p, d] Stream[b, p, d])

# Output
WO[t, d]: real
Y[p, t.] = softmax(WO[t, d] Stream[B, p, d])
```

### 12.7 Kernel SVM

```tensorlogic
# kernel_svm.tl
X[i, j]: real              # data: i=example, j=feature
Y[i]: real                 # labels
A[i]: real                 # dual variables (learned)
B: real                    # bias (learned)

# Polynomial kernel
K[i, i'] = (X[i, j] X[i', j])^n

# SVM prediction
Pred[Q] = sig(A[i] Y[i] K[Q, i] + B)
```

### 12.8 Bayesian Network

```tensorlogic
# bayesian_net.tl
# P(Cloudy), P(Rain|Cloudy), P(Sprinkler|Cloudy), P(WetGrass|Rain,Sprinkler)

CPT_Cloudy[c]: real
CPT_Rain[r, c]: real
CPT_Sprinkler[s, c]: real
CPT_WetGrass[w, r, s]: real

P_Cloudy[c] = CPT_Cloudy[c]
P_Rain[r] = CPT_Rain[r, c] P_Cloudy[c]
P_Sprinkler[s] = CPT_Sprinkler[s, c] P_Cloudy[c]
P_WetGrass[w] = CPT_WetGrass[w, r, s] P_Rain[r] P_Sprinkler[s]

# Query: P(WetGrass=1)
P_WetGrass[1]?
```

### 12.9 Embedding Space Reasoning

```tensorlogic
# embedding_reasoning.tl
# Embed relations and reason in embedding space

Emb[x, d]: real            # object embeddings (learned or random)

# Embed the Parent relation
Parent(x, y): bool
EmbParent[i, j] = Parent(x, y) Emb[x, i] Emb[y, j]

# Embed the Ancestor rules
# Ancestor(x,z) = Parent(x,z) OR Ancestor(x,y) AND Parent(y,z)
EmbAncestor[i, j] = EmbParent[i, j]
EmbAncestor[i, k] = EmbAncestor[i, j] EmbParent[j, k]

# Query in embedding space with temperature T
T: real = 0.1              # temperature (0 = deductive, >0 = analogical)
D[A, B] = sig(EmbAncestor[i, j] Emb[A, i] Emb[B, j], T)

# Extract back to Boolean
Ancestor(x, z) = step(D[x, z] - 0.5)
```

---

## 13. Implementation Phases

### Phase 1: Foundation (Core Language)
**Goal:** Parse and execute basic tensor equations

1. **Lexer**: Tokenize .tl files
2. **Parser**: Build AST for tensor equations
3. **Core types**: Tensor, Index, TensorEquation in Zig
4. **Simple interpreter**: Direct evaluation with CPU backend
5. **CLI skeleton**: `tlc run`, `tlc check`

**Milestone:** Run `Y = step(W[i] X[i])` with literal values

### Phase 2: Type System & Analysis
**Goal:** Full type/shape inference and checking at compile time

1. **Symbol table**: Track all tensors and their properties
2. **Type inference**: Infer types from usage (using Zig comptime where possible)
3. **Shape inference**: Compute output shapes from input shapes
4. **Index resolution**: Map symbolic indices to axes
5. **Error messages**: Clear diagnostics with source locations

**Milestone:** Catch shape mismatches, undefined tensors at compile time

### Phase 3: Inference Engines
**Goal:** Forward and backward chaining

1. **Dependency analysis**: Build equation dependency graph
2. **Forward chaining**: Iterative fixpoint computation
3. **Backward chaining**: Query-driven evaluation
4. **Recursion handling**: Detect and handle recursive rules

**Milestone:** Compute transitive closure (ancestor example)

### Phase 4: Backend & Performance
**Goal:** Efficient execution on GPU

1. **CUDA codegen**: Generate CUDA kernels for dense ops
2. **Metal codegen**: macOS GPU support
3. **Sparse backend**: Efficient relation storage and joins
4. **Compilation**: Fuse operations, optimize join order

**Milestone:** Run transformer attention on GPU

### Phase 5: Automatic Differentiation
**Goal:** Learn parameters via gradient descent

1. **Gradient computation**: Implement derivative rules
2. **Backward pass**: Backpropagation through tensor equations
3. **Optimizers**: SGD, Adam, etc.
4. **Training loop**: High-level training API

**Milestone:** Learn embeddings for link prediction

### Phase 6: Advanced Features
**Goal:** Full paper implementation

1. **Embedding space reasoning**: Tucker decomposition, temperature
2. **Virtual indices**: For RNNs
3. **Alternative projections**: max=, avg=, etc.
4. **Probabilistic semantics**: Graphical model support

**Milestone:** Implement transformer from paper Table 2

### Phase 7: Ecosystem
**Goal:** Production readiness

1. **REPL**: Interactive development
2. **Debugging**: Step through inference, inspect tensors
3. **LSP**: Language server for editor support
4. **Documentation**: Complete language reference

**Milestone:** User can implement novel architectures easily

---

## 14. Testing Strategy

### 14.1 Unit Tests

Using Zig's built-in test framework:

```zig
test "lexer tokenizes simple equation" {
    const source = "Y = step(W[i] X[i])";
    var lexer = Lexer.init(source);
    const tokens = try lexer.tokenize(testing.allocator);
    defer testing.allocator.free(tokens);

    try testing.expectEqual(tokens[0].type, .identifier);
    try testing.expectEqualStrings(tokens[0].lexeme, "Y");
    // ...
}
```

- **Lexer**: Token sequences for all syntax elements
- **Parser**: AST structure for representative programs
- **Type checker**: Accept valid, reject invalid programs
- **Evaluator**: Correct results for known computations

### 14.2 Integration Tests

- **Paper examples**: All examples from Section 4 of the paper
- **End-to-end**: Parse → compile → execute → verify
- **Golden tests**: Compare output against known-good results

### 14.3 Property Tests

- **Einsum equivalence**: Our joins match expected tensor contractions
- **Gradient correctness**: Compare to finite differences
- **Fixpoint convergence**: Recursive rules reach stable state

### 14.4 Benchmark Tests

- **Performance**: Track speed on standard tasks
- **Memory**: Track memory usage
- **Scaling**: Test with increasing tensor sizes

### 14.5 Reference Test Cases

From the paper:

1. **Perceptron**: `Y = step(W[i] X[i])` → correct classification
2. **MLP**: Multi-layer, learns XOR
3. **Ancestor**: Transitive closure matches naive algorithm
4. **GNN**: Node classification on simple graph
5. **Attention**: Self-attention matches reference implementation

---

## Appendix A: Key Quotes from Paper

> "The sole construct in tensor logic is the tensor equation"

> "A Datalog rule is an einsum over Boolean tensors, with a step function applied elementwise to the result"

> "The join signs are left implicit, and the projection is onto the indices on the LHS"

> "Tensor elements are 0 by default, and equations with the same LHS are implicitly summed"

> "The gradient of a tensor logic program is also a tensor logic program"

> "The most interesting feature of tensor logic is the new models it suggests [embedding space reasoning]"

---

## Appendix B: Nonlinearity Reference

| Name | Formula | Use Case |
|------|---------|----------|
| `step`, `H` | H(x) = 1 if x > 0 else 0 | Boolean logic |
| `sig`, `sigmoid` | σ(x) = 1/(1+e^(-x)) | Probability output |
| `relu` | max(0, x) | Hidden layers |
| `softmax` | e^x_i / Σe^x_j | Probability distribution |
| `lnorm` | Layer normalization | Transformers |
| `tanh` | (e^x - e^(-x))/(e^x + e^(-x)) | Bounded activation |
| `exp` | e^x | Attention scores |
| `log` | ln(x) | Cross-entropy loss |
| `sqrt` | √x | Attention scaling |
| `sin`, `cos` | Trigonometric | Positional encoding |

---

## Appendix C: Why Zig?

| Aspect | Zig | C++ | Rust |
|--------|-----|-----|------|
| Build system | Built-in, simple | CMake nightmare | Cargo (good) |
| Compile times | Fast | Slow | Slow |
| C interop | Seamless | Native | FFI overhead |
| Memory safety | Manual + checks | Manual | Borrow checker |
| Comptime | Excellent | Templates (complex) | Const generics (limited) |
| Learning curve | Moderate | Steep | Steep |
| LLVM backend | Yes | Yes | Yes |
| Error messages | Clear | Horrible | Good |

For a compiler that needs:
- Fast iteration during development
- Easy GPU library integration (CUDA, Metal)
- Compile-time tensor shape validation
- Cross-platform builds

**Zig is the best choice.**

---

*This document provides a complete specification for implementing Tensor Logic as described in Pedro Domingos' paper. The goal is to create a standalone language and toolchain that truly unifies neural and symbolic AI through the elegant simplicity of tensor equations.*
