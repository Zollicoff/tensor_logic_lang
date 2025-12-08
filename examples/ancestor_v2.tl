// Transitive closure - using domain names as indices
// This is a workaround until we implement proper variable-to-domain binding

domain i: 3
domain j: 3
domain k: 3

// Pre-declare tensors (shapes will be inferred from domain)
// Parent and Ancestor will be 3x3 matrices

// Base case: copy Parent to Ancestor
Ancestor[i,k] = Parent[i,k]

// Recursive case: transitive closure
// Ancestor[i,k] += step(Parent[i,j] Ancestor[j,k])
// This contracts over j (shared between Parent and Ancestor)
Ancestor[i,k] += step(Parent[i,j] Ancestor[j,k])
