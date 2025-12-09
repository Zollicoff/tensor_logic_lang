// Transitive closure using sparse declarations with domain binding
//
// This demonstrates proper tensor logic syntax where variables
// are bound to domains via sparse declarations.

// Define the Person domain
domain Person: 5

// Declare relations with variable-to-domain bindings
// Now x, y, z will automatically use Person's size (5)
sparse Parent(x: Person, y: Person)
sparse Ancestor(a: Person, b: Person)

// Set facts using the same variable names
// (indices inherit domain size from schema)
Parent[0,1] = 1
Parent[1,2] = 1
Parent[2,3] = 1
Parent[3,4] = 1

// Rules using any variable names - they get size from schema
Ancestor(a,b) max= step(Parent(a,b))
Ancestor(a,b) max= step(Parent(a,c) Ancestor(c,b))
