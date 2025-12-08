// Transitive closure using max= for proper Boolean semantics
//
// Family tree: 0 -> 1 -> 2 -> 3 -> 4

domain i: 5
domain j: 5
domain k: 5

// Parent facts
Parent[i,j] = 0
Parent[0,1] = 1
Parent[1,2] = 1
Parent[2,3] = 1
Parent[3,4] = 1

// Base case: direct parents are ancestors
Ancestor[i,k] max= step(Parent[i,k])

// Recursive case: transitive closure
Ancestor[i,k] max= step(Parent[i,j] Ancestor[j,k])
