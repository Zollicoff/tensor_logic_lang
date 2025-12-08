// Transitive closure test with specific facts
//
// Family tree:
//   0 (grandparent)
//   |
//   1 (parent)
//   |
//   2 (child)
//
// Parent(0,1) = true  (0 is parent of 1)
// Parent(1,2) = true  (1 is parent of 2)
//
// Expected Ancestor relations after fixpoint:
// Ancestor(0,1) = 1  (direct)
// Ancestor(1,2) = 1  (direct)
// Ancestor(0,2) = 1  (transitive: 0->1->2)

domain Person: 3

// Initialize relations to zero
Parent(x,y) = 0
Ancestor(x,z) = 0

// These would need to be set programmatically for now
// since we don't have literal tensor initialization syntax yet
// Parent[0,1] = 1
// Parent[1,2] = 1

// Base case: direct parent is an ancestor
Ancestor(x,z) = Parent(x,z)

// Recursive case: transitive closure
// This needs to run multiple times until fixpoint
Ancestor(x,z) += step(Parent(x,y) Ancestor(y,z))
