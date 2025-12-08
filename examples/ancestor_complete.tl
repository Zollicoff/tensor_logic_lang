// Complete transitive closure example with facts
//
// Family tree:
//   0 (Alice - grandparent)
//   |
//   1 (Bob - parent)
//   |
//   2 (Carol - child)
//
// Expected results:
//   Ancestor[0,1] = 1 (Alice is ancestor of Bob)
//   Ancestor[1,2] = 1 (Bob is ancestor of Carol)
//   Ancestor[0,2] = 1 (Alice is ancestor of Carol - transitive!)

// Define domains
domain i: 3
domain j: 3
domain k: 3

// Initialize Parent relation
Parent[i,j] = 0

// Set facts: Parent[x,y] = 1 means x is parent of y
Parent[0,1] = 1
Parent[1,2] = 1

// Initialize Ancestor relation
Ancestor[i,k] = 0

// Base case: direct parent is an ancestor
Ancestor[i,k] = Parent[i,k]

// Recursive case: if Parent[i,j] and Ancestor[j,k], then Ancestor[i,k]
// This is run repeatedly until fixpoint
Ancestor[i,k] += step(Parent[i,j] Ancestor[j,k])
