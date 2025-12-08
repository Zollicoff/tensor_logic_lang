// Longer ancestor chain to demonstrate fixpoint iteration
//
// Family tree: 0 -> 1 -> 2 -> 3 -> 4
// (5 generations)
//
// Expected: 10 ancestor relations total
// Direct: (0,1), (1,2), (2,3), (3,4) = 4
// Transitive: (0,2), (0,3), (0,4), (1,3), (1,4), (2,4) = 6

domain i: 5
domain j: 5
domain k: 5

// Initialize
Parent[i,j] = 0
Ancestor[i,k] = 0

// Set parent facts (chain: 0 -> 1 -> 2 -> 3 -> 4)
Parent[0,1] = 1
Parent[1,2] = 1
Parent[2,3] = 1
Parent[3,4] = 1

// Rules
Ancestor[i,k] = Parent[i,k]
Ancestor[i,k] += step(Parent[i,j] Ancestor[j,k])
