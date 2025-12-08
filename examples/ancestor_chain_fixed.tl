// Longer ancestor chain - fixed version
//
// Family tree: 0 -> 1 -> 2 -> 3 -> 4
//
// The recursive rule needs to be the ONLY rule in the fixpoint loop.
// The base case should be set ONCE before the loop.

domain i: 5
domain j: 5
domain k: 5

// Initialize Parent
Parent[i,j] = 0
Parent[0,1] = 1
Parent[1,2] = 1
Parent[2,3] = 1
Parent[3,4] = 1

// Initialize Ancestor with base case (direct parents)
Ancestor[i,k] = 0
Ancestor[i,k] = Parent[i,k]

// Recursive rule - this should run multiple times
// But the base case above resets Ancestor each iteration!
// We need += for the base case too to not clobber:
Ancestor[i,k] += step(Parent[i,j] Ancestor[j,k])
