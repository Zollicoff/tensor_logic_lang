// Family Data Module
// Defines the Parent relation for a family tree
// Export Parent for use by other modules

domain i: 5
domain j: 5
domain k: 5

// Initialize Parent relation (chain: 0 -> 1 -> 2 -> 3 -> 4)
Parent[i,j] = 0
Parent[0,1] = 1
Parent[1,2] = 1
Parent[2,3] = 1
Parent[3,4] = 1

export Parent
