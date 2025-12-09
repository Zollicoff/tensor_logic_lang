// Sparse Transitive Closure
//
// This example demonstrates sparse einsum for computing transitive closure.
// When tensors are declared with 'sparse', the interpreter uses sparse
// storage and sparse-sparse einsum, which is O(nnz^2) instead of O(n^2).
//
// For sparse relations (few edges in a large domain), sparse is more memory-efficient.

// Domain: 10 people
domain Person: 10

// Declare Parent and Ancestor as sparse relations
// This tells the interpreter to use SparseTensor storage
sparse Parent(x: Person, y: Person)
sparse Ancestor(a: Person, b: Person)

// Linear chain: 0 -> 1 -> 2 -> ... -> 9
// (Only setting first 10 for demonstration)
Parent[0,1] = 1
Parent[1,2] = 1
Parent[2,3] = 1
Parent[3,4] = 1
Parent[4,5] = 1
Parent[5,6] = 1
Parent[6,7] = 1
Parent[7,8] = 1
Parent[8,9] = 1

// Base case: direct parents are ancestors
Ancestor[a,b] max= step(Parent[a,b])

// Recursive case: transitive closure via sparse einsum
// Ancestor[a,b] = step(Parent[a,c] Ancestor[c,b])
// This contracts over c, finding all indirect ancestors
Ancestor[a,b] max= step(Parent[a,c] Ancestor[c,b])
