# Roadmap

This roadmap prioritizes work that (1) strengthens correctness and usability of the compiler, and (2) closes the biggest gaps with Domingos’ *Tensor Logic: The Language of AI* (see `docs/2510.12269v3.pdf`).

## Principles

- **Keep the language small**: prefer extending semantics of the tensor equation over adding new statement forms.
- **Correct-by-default**: shape/index mistakes should fail fast with good diagnostics.
- **One feature = end-to-end**: parser → typecheck → codegen → tests → examples.
- **Performance later, but measurable**: every “speedup” milestone should include a benchmark or regression guard.

## Status (today)

- Compiler pipeline exists: `.tl → AST → LLVM IR → clang → native`.
- Core constructs implemented: tensor equations, nonlinearities, sparse/tucker, fixpoint, autodiff, backward-chaining queries (partial).
- Tooling: working LSP + VS Code extension.

## Milestones

### M0 — Hardening & Consistency (short-term)

Goal: eliminate spec/implementation drift and reduce footguns.

- [x] **Language syntax consistency**
  - [x] Align “virtual index” spelling across docs/parser/AST.
  - [x] Accept both normalize syntaxes (`i.` and `/i`) and keep printed form consistent.
  - [x] Clarify boolean-vs-real tensor syntax (`()` vs `[]`) in docs/examples.
- [x] **Better CLI/LSP diagnostics**
  - [x] Add actionable error messages for missing `clang` or bad paths.
  - [x] LSP: tighten document lifecycle correctness and version tracking.
- [x] **Examples as smoke tests**
  - [x] Expand integration coverage across multiple examples (build-and-run + check-only).
- [ ] **Data ingestion (paper-practical)**
  - [x] Document binary save/load format and NumPy interop in README.
  - [ ] *Deferred to M1.5+*: Text/CSV ingestion requires parsing in LLVM IR or linking C stdlib—complex.

Deliverable: a stable release with fewer surprises and clearer docs.

---

### M1 — Basic Shape & Index Checks (high leverage)

Goal: make tensor programs *meaningfully type-checked*, not just "boolean vs real", without taking on full global shape inference immediately.

- [x] **Rank checking**
  - [x] Ensure tensor references consistently use the same rank across a program.
  - [ ] Ensure LHS/RHS ranks match where required (deferred to M1.1 with domain tracking).
- [x] **Index symbol hygiene**
  - [x] Validate arithmetic indices (e.g., `i/2`) are well-defined (division by zero = error).
  - [ ] Detect inconsistent reuse of the same index name in incompatible positions (deferred to M1.1).
- [x] **Better errors**
  - [x] Point to exact tensor/index that caused mismatch and show expected vs found.
  - [x] Show severity (error/warning), cross-reference original definition location.

Deliverable: a release where most silent-wrong programs become compile errors.

---

### M1.5 — Backward-Chaining Queries (paper-core)

Goal: make backward chaining a complete, testable inference mode (not just a partial codegen path).

- [x] **Query semantics**
  - [x] Document how queries interact with forward-mode tensors in README.
  - [x] Clarify the model for recursion:
    - Forward mode uses fixpoint iteration for global closure.
    - Backward mode uses recursive evaluation + memoization for demand-driven queries.
  - [x] Document which operations are forward-only (softmax, lnorm, concat, avg=).
- [ ] **Codegen completeness**
  - [x] Basic contracted indices work in backward mode.
  - [x] Basic nonlinearities (relu, sigmoid, step, tanh, exp, log, sqrt, abs) implemented.
  - [ ] *Known issue*: Some recursive patterns produce incorrect results at runtime.
  - [ ] *Deferred*: primed indices, div indices, virtual indices in backward mode.
- [x] **Tests**
  - [x] Add backward_vs_forward.tl example demonstrating both modes.
  - [x] Backward query examples pass type checking (check_only mode).
  - [ ] *Deferred*: Runtime comparison tests pending codegen fixes.

Deliverable: backward-chaining mode that users can rely on for "query only what's needed."

**Status**: Infrastructure complete, documentation added. Runtime correctness issues remain for complex recursive patterns.

---

### M1.1 — Domain Tracking & Shape Inference (bigger compiler work)

Goal: add domain-aware shape inference once basic checks are solid.

- [x] **Domain-aware index checking**
  - [x] Track domain for each index symbol within a scope and reject inconsistent reuse.
  - [x] Warn when index domain is inferred from name (not tensor position).
  - [x] Warn when same index used with different domain names (even if sizes match).
  - [ ] *Deferred*: Infer missing domain sizes (currently falls back to default 10).
- [x] **Tensor shape inference**
  - [x] Infer tensor shapes from sparse/tucker declarations + domain declarations.
  - [x] Propagate shape requirements through RHS tensor references (validates index consistency).
- [x] **Better errors (shape-centric)**
  - [x] Report expected vs found sizes/domains with cross-reference locations.

**Status**: Core index consistency checking complete.
- **Error**: Same index symbol used with different domain sizes.
- **Warning**: Index domain inferred from name (name-based coupling made visible).
- **Warning**: Same index used with different domain names (sizes match but semantics may differ).

---

### M2 — Semiring / Aggregation Semantics (paper alignment)

Goal: make `+=`, `max=`, `min=`, `avg=` semantics first-class and predictable (logic/prob/neural unification).

- [ ] **Define aggregation rules precisely**
  - [ ] Formalize how contracted indices aggregate (sum/max/min/avg) for each accumulation operator.
  - [ ] Make fixpoint iteration semantics depend on monotonic operators when applicable.
- [ ] **Operator-accurate codegen**
  - [ ] Ensure “assign with contraction” behavior is clearly specified and tested.
  - [ ] Add tests for `max=` / `min=` / `avg=` on both dense and recursive programs.

Deliverable: a release with paper-faithful aggregation behavior.

---

### M3 — Temperature-Gated Reasoning (paper alignment)

Goal: expose the paper’s “deduction ↔ analogy” control knob.

- [ ] **Temperature sigmoid**
  - [ ] Language: `sigmoid(x, T)` is already parsed; ensure it is supported end-to-end and documented as the reasoning dial.
  - [ ] Document recommended patterns for “strict logic” vs “soft/analogical” regimes.
- [ ] **Stability tests**
  - [ ] Add tests/examples showing behavior changes smoothly with temperature.

Deliverable: a release with a usable, documented temperature mechanism.

---

### M4 — Embedding-Space / Tucker-Backed Reasoning (paper alignment)

Goal: implement a *practical* subset of the paper’s embedding-space inference story.

- [ ] **Tucker as an explicit execution strategy (start manual)**
  - [ ] Start with manual `tucker` declarations and well-defined lowering.
  - [ ] Avoid “automatic lowering” until semantics, costs, and heuristics are clear.
  - [ ] Implement execution on decomposed representation for a narrow, testable subset.
- [ ] **Approximate membership / retrieval**
  - [ ] Provide a minimal set of operations enabling “similarity-based” reasoning.
  - [ ] Add example(s) demonstrating approximate inference and how/when to re-threshold.
  - [ ] Add a testing strategy for approximate behavior (e.g., statistical/threshold-based assertions).

Deliverable: a release that demonstrates the paper’s novel scaling idea in a small, testable way.

---

### M5 — Performance & GPU Path (medium/long-term)

Goal: accelerate dense workloads (attention/GNN) without destabilizing semantics.

- [ ] **Benchmarks**
  - [ ] Add a tiny benchmark harness (compile + run time) for a few canonical programs.
- [ ] **GPU backend**
  - [ ] Decide scope: CUDA vs Metal first.
  - [ ] Start with a narrow set of kernels (matmul/einsum subset, softmax) and expand.

Deliverable: measurable speedups on supported ops.

## Non-goals (for now)

- Full Prolog unification/function symbols as built-ins (paper suggests implementing within TL if desired).
- Large-scale distributed execution.
- A large standard library—keep the core tight until semantics stabilize.

## How to contribute

- Prefer PRs that include: (1) a minimal `.tl` example, (2) compiler changes, (3) a test (unit/integration), and (4) doc updates.
- Keep features behind clear syntax and avoid adding new statement forms unless necessary.

## Notes

- **File I/O**: the language already includes `save`/`load`; extending this toward the paper’s “read data into tensors” story should be driven by concrete examples and tests.
- **In-flight work**: if the repo has local changes (e.g., LSP fixes), fold them into M0 with a small regression test where possible.
