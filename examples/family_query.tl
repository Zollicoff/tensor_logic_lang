// Family Query Module
// Imports the Parent relation and computes Ancestor

import "examples/family_data.tl"

// Define Ancestor using imported Parent
Ancestor[i,k] max= step(Parent[i,k])
Ancestor[i,k] max= step(Parent[i,j] Ancestor[j,k])
