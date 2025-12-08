// Transitive closure test using sparse declarations
//
// Family tree:
//   0 (grandparent)
//   |
//   1 (parent)
//   |
//   2 (child)

domain Person: 3

// Declare relations with domain bindings
sparse Parent(x: Person, y: Person)
sparse Ancestor(x: Person, z: Person)

// Base case: direct parent is an ancestor
Ancestor(x,z) = Parent(x,z)

// Recursive case: transitive closure
Ancestor(x,z) += step(Parent(x,y) Ancestor(y,z))
