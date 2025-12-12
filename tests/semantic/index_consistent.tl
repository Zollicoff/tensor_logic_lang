// Test: Index used consistently with same domain - should pass
// y is contracted in the ancestor relation, used consistently

domain p: 10
sparse Parent[p, p]

// Ancestor relation (transitive closure)
// y appears in both Ancestor[x,y] and Parent[y,z] - both from domain p
Ancestor[x,y] = step(Parent[x,y])
Ancestor[x,z] max= step(Ancestor[x,y] Parent[y,z])
