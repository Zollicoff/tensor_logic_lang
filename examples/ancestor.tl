// Transitive closure (ancestor relation)
// From Section 3 of Pedro Domingos' Tensor Logic paper

// Base case: direct parent
Ancestor(x,z) = Parent(x,z)

// Recursive case: if Parent(x,y) and Ancestor(y,z), then Ancestor(x,z)
Ancestor(x,z) += step(Parent(x,y) Ancestor(y,z))
