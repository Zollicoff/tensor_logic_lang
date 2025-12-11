// Transitive Closure (Ancestor Relation)
//
// Classic logic programming example: computing reachability
// From Section 3 of Pedro Domingos' "Tensor Logic" paper
//
// Given a Parent relation, compute all Ancestor pairs

domain p: 10

// Parent relation (sparse - only a few edges)
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

// Ancestor relation (transitive closure of Parent)
// Base case: direct parents are ancestors
Ancestor[x,y] = step(Parent[x,y])

// Recursive case: if Parent[x,z] and Ancestor[z,y], then Ancestor[x,y]
// Uses max= for fixpoint iteration (converges when no new ancestors found)
Ancestor[x,y] max= step(Parent[x,z] Ancestor[z,y])
