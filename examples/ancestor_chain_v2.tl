// Longer ancestor chain - accumulation only version
//
// Family tree: 0 -> 1 -> 2 -> 3 -> 4

domain i: 5
domain j: 5
domain k: 5

// Initialize Parent (element assignments don't reset)
Parent[i,j] = 0
Parent[0,1] = 1
Parent[1,2] = 1
Parent[2,3] = 1
Parent[3,4] = 1

// Initialize Ancestor to zero
Ancestor[i,k] = 0

// Use += for base case so it accumulates, not resets
Ancestor[i,k] += Parent[i,k]

// Recursive rule
Ancestor[i,k] += step(Parent[i,j] Ancestor[j,k])
