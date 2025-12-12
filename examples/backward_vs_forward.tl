// Example: Forward and backward query modes
//
// Forward mode: Tensor? (no constant indices)
//   - Computes entire tensor via nested loops
//   - Uses fixpoint iteration for recursive equations
//
// Backward mode: Tensor[0,5]? (with constant indices)
//   - Computes only queried cell via recursive functions
//   - Uses memoization to avoid redundant computation
//
// NOTE: Backward mode is currently limited. Some patterns may not
// produce correct results. See ROADMAP.MD for planned improvements.

domain p: 10

// Parent relation (sparse)
sparse Parent[p, p]

// Linear chain: 0 -> 1 -> 2 -> ... -> 9
Parent[0,1] = 1
Parent[1,2] = 1
Parent[2,3] = 1
Parent[3,4] = 1
Parent[4,5] = 1
Parent[5,6] = 1
Parent[6,7] = 1
Parent[7,8] = 1
Parent[8,9] = 1

// Ancestor relation (transitive closure)
Ancestor[x,y] = step(Parent[x,y])
Ancestor[x,z] max= step(Ancestor[x,y] Parent[y,z])

// Forward query: compute all ancestor pairs
Ancestor?

// Backward queries: compute specific pairs only
// These use recursive evaluation with memoization
Ancestor[0, 5]?
Ancestor[0, 9]?
Ancestor[5, 2]?
