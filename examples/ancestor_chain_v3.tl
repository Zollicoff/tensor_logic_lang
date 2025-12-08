// Longer ancestor chain - avoiding reset in loop
//
// Family tree: 0 -> 1 -> 2 -> 3 -> 4
//
// Key insight: Don't use = for Ancestor, only +=
// The tensor starts as zeros anyway

domain i: 5
domain j: 5
domain k: 5

// Parent facts (these are idempotent with =)
Parent[i,j] = 0
Parent[0,1] = 1
Parent[1,2] = 1
Parent[2,3] = 1
Parent[3,4] = 1

// DON'T initialize Ancestor - it starts as zeros
// DON'T use = which would reset each iteration

// Base case: add direct parents (idempotent with step)
Ancestor[i,k] += step(Parent[i,k])

// Recursive case
Ancestor[i,k] += step(Parent[i,j] Ancestor[j,k])
