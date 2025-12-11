// Backward Chaining Query
//
// Demonstrates demand-driven inference: only compute what's needed.
// Instead of computing all Ancestor pairs, query specific ones.
//
// Forward chaining: compute entire Ancestor tensor (O(n²) work)
// Backward chaining: compute only queried pairs (O(path length) work)

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

// Ancestor relation (defined recursively)
Ancestor[x,y] = step(Parent[x,y])
Ancestor[x,z] max= step(Parent[x,y] Ancestor[y,z])

// Backward chaining queries - only compute these specific pairs
// Instead of computing all 100 pairs, we only compute what we need
// Queries with constant indices automatically use backward chaining
Ancestor[0, 5]?
Ancestor[0, 9]?
Ancestor[5, 2]?
