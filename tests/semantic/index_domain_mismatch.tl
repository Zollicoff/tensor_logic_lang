// Test: Index used with different domains that have same size
// This should produce a WARNING (not error) since sizes match but semantics may differ

domain A: 5
domain B: 5

sparse X[A]
sparse Y[B]

// i is used in X (domain A, size 5) and Y (domain B, size 5)
// Sizes match, but domain names differ - this is suspicious
Z[i] = X[i] Y[i]
