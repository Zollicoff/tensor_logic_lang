// Transitive closure test with matching domain/index names
//
// Using domain name 'p' for both domain and index

domain p: 3

// Relations using domain-named indices
Parent(p,p) = 0
Ancestor(p,p) = 0

// The actual rules
Ancestor(p,p) = Parent(p,p)
Ancestor(p,p) += step(Parent(p,p) Ancestor(p,p))
