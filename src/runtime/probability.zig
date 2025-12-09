// Probabilistic Inference Module for Tensor Logic
//
// From Domingos' paper: Tensor logic enables probabilistic reasoning through
// weighted model counting and belief propagation. This module implements:
//
// 1. Weighted Model Counting (WMC) - compute partition functions
// 2. Marginal inference - P(X) from joint distributions
// 3. Belief propagation - message passing on factor graphs
// 4. Sampling - generate samples from distributions
//

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const DenseTensor = tensor_mod.DenseTensor;

/// A factor in a factor graph - represents P(X1, X2, ..., Xn)
pub const Factor = struct {
    /// Variable indices this factor depends on
    variables: []const usize,
    /// Potential values (unnormalized probabilities)
    potentials: *DenseTensor(f64),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, variables: []const usize, potentials: *DenseTensor(f64)) !Factor {
        const vars_copy = try allocator.alloc(usize, variables.len);
        @memcpy(vars_copy, variables);
        return Factor{
            .variables = vars_copy,
            .potentials = potentials,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Factor) void {
        self.allocator.free(self.variables);
    }
};

/// Factor graph for probabilistic inference
pub const FactorGraph = struct {
    /// Number of variables
    num_vars: usize,
    /// Domain size for each variable (assumed binary if not specified)
    var_domains: []usize,
    /// Factors in the graph
    factors: std.ArrayListUnmanaged(Factor),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_vars: usize) !FactorGraph {
        const domains = try allocator.alloc(usize, num_vars);
        @memset(domains, 2); // Default to binary
        return FactorGraph{
            .num_vars = num_vars,
            .var_domains = domains,
            .factors = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FactorGraph) void {
        for (self.factors.items) |*f| {
            f.deinit();
        }
        self.factors.deinit(self.allocator);
        self.allocator.free(self.var_domains);
    }

    pub fn addFactor(self: *FactorGraph, variables: []const usize, potentials: *DenseTensor(f64)) !void {
        const factor = try Factor.init(self.allocator, variables, potentials);
        try self.factors.append(self.allocator, factor);
    }

    pub fn setDomain(self: *FactorGraph, var_idx: usize, domain_size: usize) void {
        if (var_idx < self.var_domains.len) {
            self.var_domains[var_idx] = domain_size;
        }
    }
};

/// Weighted Model Counting
/// Computes Z = sum over all assignments of product of factor potentials
pub fn weightedModelCount(graph: *const FactorGraph) f64 {
    // For small problems, enumerate all assignments
    const total_assignments = blk: {
        var n: usize = 1;
        for (graph.var_domains) |d| {
            n *= d;
        }
        break :blk n;
    };

    var z: f64 = 0;

    // Enumerate all assignments
    var assignment: usize = 0;
    while (assignment < total_assignments) : (assignment += 1) {
        // Decode assignment to variable values
        var weight: f64 = 1.0;

        for (graph.factors.items) |factor| {
            // Compute index into factor's potential table
            var factor_idx: usize = 0;
            var multiplier: usize = 1;

            var i = factor.variables.len;
            while (i > 0) {
                i -= 1;
                const var_idx = factor.variables[i];
                const var_val = getVarValue(assignment, var_idx, graph.var_domains);
                factor_idx += var_val * multiplier;
                multiplier *= graph.var_domains[var_idx];
            }

            if (factor_idx < factor.potentials.data.len) {
                weight *= factor.potentials.data[factor_idx];
            }
        }

        z += weight;
    }

    return z;
}

/// Extract value of variable from assignment encoding
fn getVarValue(assignment: usize, var_idx: usize, domains: []const usize) usize {
    var remaining = assignment;
    for (0..var_idx) |i| {
        remaining /= domains[i];
    }
    return remaining % domains[var_idx];
}

/// Compute marginal probability P(X_i = x)
pub fn marginal(allocator: std.mem.Allocator, graph: *const FactorGraph, var_idx: usize) !DenseTensor(f64) {
    const domain_size = graph.var_domains[var_idx];
    var result = try DenseTensor(f64).init(allocator, &[_]usize{domain_size});

    // Enumerate all assignments, accumulate by variable value
    const total_assignments = blk: {
        var n: usize = 1;
        for (graph.var_domains) |d| {
            n *= d;
        }
        break :blk n;
    };

    var assignment: usize = 0;
    while (assignment < total_assignments) : (assignment += 1) {
        var weight: f64 = 1.0;

        for (graph.factors.items) |factor| {
            var factor_idx: usize = 0;
            var multiplier: usize = 1;

            var i = factor.variables.len;
            while (i > 0) {
                i -= 1;
                const v_idx = factor.variables[i];
                const var_val = getVarValue(assignment, v_idx, graph.var_domains);
                factor_idx += var_val * multiplier;
                multiplier *= graph.var_domains[v_idx];
            }

            if (factor_idx < factor.potentials.data.len) {
                weight *= factor.potentials.data[factor_idx];
            }
        }

        const var_val = getVarValue(assignment, var_idx, graph.var_domains);
        result.data[var_val] += weight;
    }

    // Normalize
    var sum: f64 = 0;
    for (result.data) |v| sum += v;
    if (sum > 0) {
        for (result.data) |*v| v.* /= sum;
    }

    return result;
}

/// Belief Propagation Message
pub const Message = struct {
    from: usize, // factor or variable index
    to: usize,
    values: []f64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, from: usize, to: usize, size: usize) !Message {
        const values = try allocator.alloc(f64, size);
        @memset(values, 1.0); // Initialize to uniform
        return Message{
            .from = from,
            .to = to,
            .values = values,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Message) void {
        self.allocator.free(self.values);
    }
};

/// Loopy Belief Propagation for approximate inference
pub const BeliefPropagation = struct {
    graph: *const FactorGraph,
    /// Messages from variables to factors
    var_to_factor: std.AutoHashMap(struct { usize, usize }, Message),
    /// Messages from factors to variables
    factor_to_var: std.AutoHashMap(struct { usize, usize }, Message),
    allocator: std.mem.Allocator,
    /// Convergence threshold
    threshold: f64,
    /// Maximum iterations
    max_iters: usize,

    pub fn init(allocator: std.mem.Allocator, graph: *const FactorGraph) BeliefPropagation {
        return BeliefPropagation{
            .graph = graph,
            .var_to_factor = std.AutoHashMap(struct { usize, usize }, Message).init(allocator),
            .factor_to_var = std.AutoHashMap(struct { usize, usize }, Message).init(allocator),
            .allocator = allocator,
            .threshold = 1e-6,
            .max_iters = 100,
        };
    }

    pub fn deinit(self: *BeliefPropagation) void {
        var v2f_iter = self.var_to_factor.valueIterator();
        while (v2f_iter.next()) |msg| {
            var m = msg.*;
            m.deinit();
        }
        self.var_to_factor.deinit();

        var f2v_iter = self.factor_to_var.valueIterator();
        while (f2v_iter.next()) |msg| {
            var m = msg.*;
            m.deinit();
        }
        self.factor_to_var.deinit();
    }

    /// Initialize messages
    pub fn initMessages(self: *BeliefPropagation) !void {
        for (self.graph.factors.items, 0..) |factor, f_idx| {
            for (factor.variables) |v_idx| {
                const domain = self.graph.var_domains[v_idx];

                // Variable to factor message
                const v2f = try Message.init(self.allocator, v_idx, f_idx, domain);
                try self.var_to_factor.put(.{ v_idx, f_idx }, v2f);

                // Factor to variable message
                const f2v = try Message.init(self.allocator, f_idx, v_idx, domain);
                try self.factor_to_var.put(.{ f_idx, v_idx }, f2v);
            }
        }
    }

    /// Run belief propagation
    pub fn run(self: *BeliefPropagation) !usize {
        try self.initMessages();

        var iter: usize = 0;
        while (iter < self.max_iters) : (iter += 1) {
            var max_change: f64 = 0;

            // Update variable to factor messages
            for (self.graph.factors.items, 0..) |factor, f_idx| {
                for (factor.variables) |v_idx| {
                    const change = try self.updateVarToFactor(v_idx, f_idx);
                    max_change = @max(max_change, change);
                }
            }

            // Update factor to variable messages
            for (self.graph.factors.items, 0..) |factor, f_idx| {
                for (factor.variables) |v_idx| {
                    const change = try self.updateFactorToVar(f_idx, v_idx);
                    max_change = @max(max_change, change);
                }
            }

            if (max_change < self.threshold) {
                return iter + 1;
            }
        }

        return self.max_iters;
    }

    /// Update message from variable to factor
    fn updateVarToFactor(self: *BeliefPropagation, v_idx: usize, target_f: usize) !f64 {
        const domain = self.graph.var_domains[v_idx];
        const new_msg = try self.allocator.alloc(f64, domain);
        defer self.allocator.free(new_msg);

        @memset(new_msg, 1.0);

        // Product of incoming messages from other factors
        for (self.graph.factors.items, 0..) |factor, f_idx| {
            if (f_idx == target_f) continue;

            // Check if this factor connects to v_idx
            var connects = false;
            for (factor.variables) |v| {
                if (v == v_idx) {
                    connects = true;
                    break;
                }
            }

            if (connects) {
                if (self.factor_to_var.get(.{ f_idx, v_idx })) |msg| {
                    for (new_msg, 0..) |*m, i| {
                        if (i < msg.values.len) {
                            m.* *= msg.values[i];
                        }
                    }
                }
            }
        }

        // Normalize
        var sum: f64 = 0;
        for (new_msg) |v| sum += v;
        if (sum > 0) {
            for (new_msg) |*v| v.* /= sum;
        }

        // Compute change and update
        var max_change: f64 = 0;
        if (self.var_to_factor.getPtr(.{ v_idx, target_f })) |msg| {
            for (msg.values, 0..) |*old, i| {
                if (i < new_msg.len) {
                    max_change = @max(max_change, @abs(old.* - new_msg[i]));
                    old.* = new_msg[i];
                }
            }
        }

        return max_change;
    }

    /// Update message from factor to variable
    fn updateFactorToVar(self: *BeliefPropagation, f_idx: usize, target_v: usize) !f64 {
        const factor = self.graph.factors.items[f_idx];
        const domain = self.graph.var_domains[target_v];
        var new_msg = try self.allocator.alloc(f64, domain);
        defer self.allocator.free(new_msg);

        @memset(new_msg, 0.0);

        // Sum over all configurations of other variables
        const num_other_vars = factor.variables.len - 1;
        if (num_other_vars == 0) {
            // Single variable factor - just copy potentials
            for (new_msg, 0..) |*m, i| {
                if (i < factor.potentials.data.len) {
                    m.* = factor.potentials.data[i];
                }
            }
        } else {
            // Marginalize over other variables
            const total_configs = blk: {
                var n: usize = 1;
                for (factor.variables) |v| {
                    if (v != target_v) {
                        n *= self.graph.var_domains[v];
                    }
                }
                break :blk n;
            };

            for (0..domain) |target_val| {
                var config: usize = 0;
                while (config < total_configs) : (config += 1) {
                    // Compute potential index and message product
                    var potential_idx: usize = 0;
                    var multiplier: usize = 1;
                    var msg_product: f64 = 1.0;
                    var other_config = config;

                    var i = factor.variables.len;
                    while (i > 0) {
                        i -= 1;
                        const v = factor.variables[i];
                        var val: usize = undefined;

                        if (v == target_v) {
                            val = target_val;
                        } else {
                            val = other_config % self.graph.var_domains[v];
                            other_config /= self.graph.var_domains[v];

                            // Multiply by incoming message
                            if (self.var_to_factor.get(.{ v, f_idx })) |msg| {
                                if (val < msg.values.len) {
                                    msg_product *= msg.values[val];
                                }
                            }
                        }

                        potential_idx += val * multiplier;
                        multiplier *= self.graph.var_domains[v];
                    }

                    if (potential_idx < factor.potentials.data.len) {
                        new_msg[target_val] += factor.potentials.data[potential_idx] * msg_product;
                    }
                }
            }
        }

        // Normalize
        var sum: f64 = 0;
        for (new_msg) |v| sum += v;
        if (sum > 0) {
            for (new_msg) |*v| v.* /= sum;
        }

        // Compute change and update
        var max_change: f64 = 0;
        if (self.factor_to_var.getPtr(.{ f_idx, target_v })) |msg| {
            for (msg.values, 0..) |*old, i| {
                if (i < new_msg.len) {
                    max_change = @max(max_change, @abs(old.* - new_msg[i]));
                    old.* = new_msg[i];
                }
            }
        }

        return max_change;
    }

    /// Get belief (marginal) for a variable after running BP
    pub fn getBelief(self: *BeliefPropagation, v_idx: usize) !DenseTensor(f64) {
        const domain = self.graph.var_domains[v_idx];
        const belief = try DenseTensor(f64).init(self.allocator, &[_]usize{domain});

        // Product of all incoming messages
        @memset(belief.data, 1.0);

        for (self.graph.factors.items, 0..) |factor, f_idx| {
            var connects = false;
            for (factor.variables) |v| {
                if (v == v_idx) {
                    connects = true;
                    break;
                }
            }

            if (connects) {
                if (self.factor_to_var.get(.{ f_idx, v_idx })) |msg| {
                    for (belief.data, 0..) |*b, i| {
                        if (i < msg.values.len) {
                            b.* *= msg.values[i];
                        }
                    }
                }
            }
        }

        // Normalize
        var sum: f64 = 0;
        for (belief.data) |v| sum += v;
        if (sum > 0) {
            for (belief.data) |*v| v.* /= sum;
        }

        return belief;
    }
};

/// Sample from a distribution tensor (assumes 1D normalized)
pub fn sample(rng: std.Random, dist: *const DenseTensor(f64)) usize {
    const r = rng.float(f64);
    var cumsum: f64 = 0;
    for (dist.data, 0..) |p, i| {
        cumsum += p;
        if (r < cumsum) return i;
    }
    return dist.data.len - 1;
}

/// Gibbs sampling from a factor graph
pub fn gibbsSample(
    allocator: std.mem.Allocator,
    graph: *const FactorGraph,
    num_samples: usize,
    burn_in: usize,
    rng: std.Random,
) ![][]usize {
    const samples = try allocator.alloc([]usize, num_samples);

    // Initialize random assignment
    const current = try allocator.alloc(usize, graph.num_vars);
    defer allocator.free(current);
    for (current, 0..) |*c, i| {
        c.* = rng.intRangeLessThan(usize, 0, graph.var_domains[i]);
    }

    // Burn-in
    for (0..burn_in) |_| {
        try gibbsStep(allocator, graph, current, rng);
    }

    // Collect samples
    for (0..num_samples) |s| {
        try gibbsStep(allocator, graph, current, rng);
        samples[s] = try allocator.alloc(usize, graph.num_vars);
        @memcpy(samples[s], current);
    }

    return samples;
}

/// Single Gibbs sampling step
fn gibbsStep(allocator: std.mem.Allocator, graph: *const FactorGraph, assignment: []usize, rng: std.Random) !void {
    for (0..graph.num_vars) |v_idx| {
        const domain = graph.var_domains[v_idx];
        var probs = try allocator.alloc(f64, domain);
        defer allocator.free(probs);

        // Compute conditional distribution
        for (0..domain) |val| {
            assignment[v_idx] = val;
            var prob: f64 = 1.0;

            for (graph.factors.items) |factor| {
                // Check if factor involves this variable
                var involves = false;
                for (factor.variables) |v| {
                    if (v == v_idx) {
                        involves = true;
                        break;
                    }
                }

                if (involves) {
                    // Compute factor value
                    var idx: usize = 0;
                    var mult: usize = 1;
                    var i = factor.variables.len;
                    while (i > 0) {
                        i -= 1;
                        const v = factor.variables[i];
                        idx += assignment[v] * mult;
                        mult *= graph.var_domains[v];
                    }
                    if (idx < factor.potentials.data.len) {
                        prob *= factor.potentials.data[idx];
                    }
                }
            }
            probs[val] = prob;
        }

        // Normalize and sample
        var sum: f64 = 0;
        for (probs) |p| sum += p;
        if (sum > 0) {
            for (probs) |*p| p.* /= sum;
        }

        // Sample from distribution
        const r = rng.float(f64);
        var cumsum: f64 = 0;
        for (probs, 0..) |p, val| {
            cumsum += p;
            if (r < cumsum) {
                assignment[v_idx] = val;
                break;
            }
        }
    }
}

/// Normalize a tensor to sum to 1 (in-place)
pub fn normalize(tensor: *DenseTensor(f64)) void {
    var sum: f64 = 0;
    for (tensor.data) |v| sum += v;
    if (sum > 0) {
        for (tensor.data) |*v| v.* /= sum;
    }
}

/// Softmax normalization (in-place)
pub fn softmax(tensor: *DenseTensor(f64)) void {
    // Find max for numerical stability
    var max_val: f64 = tensor.data[0];
    for (tensor.data) |v| {
        if (v > max_val) max_val = v;
    }

    // Compute exp and sum
    var sum: f64 = 0;
    for (tensor.data) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }

    // Normalize
    if (sum > 0) {
        for (tensor.data) |*v| v.* /= sum;
    }
}

/// Compute log partition function (log Z)
pub fn logPartitionFunction(graph: *const FactorGraph) f64 {
    return @log(weightedModelCount(graph));
}

/// Entropy of a distribution
pub fn entropy(dist: *const DenseTensor(f64)) f64 {
    var h: f64 = 0;
    for (dist.data) |p| {
        if (p > 0) {
            h -= p * @log(p);
        }
    }
    return h;
}

/// KL divergence D(P || Q)
pub fn klDivergence(p: *const DenseTensor(f64), q: *const DenseTensor(f64)) f64 {
    var kl: f64 = 0;
    for (p.data, 0..) |pi, i| {
        if (pi > 0 and i < q.data.len and q.data[i] > 0) {
            kl += pi * (@log(pi) - @log(q.data[i]));
        }
    }
    return kl;
}

// ============================================================================
// Tests
// ============================================================================

test "weighted model count simple" {
    const allocator = std.testing.allocator;

    // Simple 2-variable model: P(X, Y) = phi(X) * phi(Y)
    var graph = try FactorGraph.init(allocator, 2);
    defer graph.deinit();

    // Factor on X: [0.3, 0.7] (P(X=0)=0.3, P(X=1)=0.7)
    var phi_x = try DenseTensor(f64).init(allocator, &[_]usize{2});
    defer phi_x.deinit();
    phi_x.data[0] = 0.3;
    phi_x.data[1] = 0.7;

    // Factor on Y: [0.4, 0.6]
    var phi_y = try DenseTensor(f64).init(allocator, &[_]usize{2});
    defer phi_y.deinit();
    phi_y.data[0] = 0.4;
    phi_y.data[1] = 0.6;

    try graph.addFactor(&[_]usize{0}, &phi_x);
    try graph.addFactor(&[_]usize{1}, &phi_y);

    const z = weightedModelCount(&graph);
    // Z = 0.3*0.4 + 0.3*0.6 + 0.7*0.4 + 0.7*0.6 = 1.0
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), z, 0.0001);
}

test "marginal computation" {
    const allocator = std.testing.allocator;

    var graph = try FactorGraph.init(allocator, 2);
    defer graph.deinit();

    // Joint distribution as single factor
    var joint = try DenseTensor(f64).init(allocator, &[_]usize{ 2, 2 });
    defer joint.deinit();
    // P(X=0,Y=0)=0.1, P(X=0,Y=1)=0.2, P(X=1,Y=0)=0.3, P(X=1,Y=1)=0.4
    joint.data[0] = 0.1; // X=0, Y=0
    joint.data[1] = 0.2; // X=0, Y=1
    joint.data[2] = 0.3; // X=1, Y=0
    joint.data[3] = 0.4; // X=1, Y=1

    try graph.addFactor(&[_]usize{ 0, 1 }, &joint);

    // Marginal P(X)
    var marginal_x = try marginal(allocator, &graph, 0);
    defer marginal_x.deinit();

    // P(X=0) = 0.1+0.2 = 0.3, P(X=1) = 0.3+0.4 = 0.7
    try std.testing.expectApproxEqAbs(@as(f64, 0.3), marginal_x.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), marginal_x.data[1], 0.0001);
}

test "belief propagation simple" {
    const allocator = std.testing.allocator;

    var graph = try FactorGraph.init(allocator, 2);
    defer graph.deinit();

    // Independent factors
    var phi_x = try DenseTensor(f64).init(allocator, &[_]usize{2});
    defer phi_x.deinit();
    phi_x.data[0] = 0.3;
    phi_x.data[1] = 0.7;

    var phi_y = try DenseTensor(f64).init(allocator, &[_]usize{2});
    defer phi_y.deinit();
    phi_y.data[0] = 0.4;
    phi_y.data[1] = 0.6;

    try graph.addFactor(&[_]usize{0}, &phi_x);
    try graph.addFactor(&[_]usize{1}, &phi_y);

    var bp = BeliefPropagation.init(allocator, &graph);
    defer bp.deinit();

    _ = try bp.run();

    var belief_x = try bp.getBelief(0);
    defer belief_x.deinit();

    // Should match marginals
    try std.testing.expectApproxEqAbs(@as(f64, 0.3), belief_x.data[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), belief_x.data[1], 0.01);
}

test "entropy computation" {
    const allocator = std.testing.allocator;

    // Uniform distribution has max entropy
    var uniform = try DenseTensor(f64).init(allocator, &[_]usize{2});
    defer uniform.deinit();
    uniform.data[0] = 0.5;
    uniform.data[1] = 0.5;

    const h = entropy(&uniform);
    // H = -0.5*log(0.5) - 0.5*log(0.5) = log(2) ≈ 0.693
    try std.testing.expectApproxEqAbs(@as(f64, 0.693147), h, 0.001);
}
