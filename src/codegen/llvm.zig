// LLVM IR Code Generator for Tensor Logic
//
// Main orchestrator for LLVM code generation. Imports specialized modules:
// - types.zig:    Shared data structures (TensorInfo, IndexVar)
// - tensor.zig:   Tensor allocation and indexing
// - expr.zig:     Expression evaluation
// - einsum.zig:   Einstein summation loop generation
// - softmax.zig:  Softmax with reduction
// - fixpoint.zig: Recursive equation handling
//
// Compile output with: clang output.ll -o program -lm

const std = @import("std");
const ast = @import("../frontend/ast.zig");

// Import codegen modules
const types = @import("types.zig");
const tensor_mod = @import("tensor.zig");
const expr_mod = @import("expr.zig");
const einsum = @import("einsum.zig");
const softmax = @import("softmax.zig");
const layernorm = @import("layernorm.zig");
const fixpoint = @import("fixpoint.zig");
const autodiff = @import("autodiff.zig");
const concat = @import("concat.zig");
const backward = @import("backward.zig");
const sparse = @import("sparse.zig");
const tucker = @import("tucker.zig");

// Re-export types
pub const TensorInfo = types.TensorInfo;
pub const IndexVar = types.IndexVar;

/// LLVM IR emitter - the main codegen context
pub const LLVMCodegen = struct {
    allocator: std.mem.Allocator,
    output: std.ArrayListUnmanaged(u8),

    // Arena for temporary strings (temps, labels, etc.)
    string_arena: std.heap.ArenaAllocator,

    // Counters for unique names
    temp_counter: usize,
    label_counter: usize,
    equation_counter: usize,

    // Track tensors and domains
    tensors: std.StringHashMapUnmanaged(TensorInfo),
    domains: std.StringHashMapUnmanaged(usize),

    // Pre-computed max dimensions for tensors (from scanning all constant indices)
    tensor_max_dims: std.StringHashMapUnmanaged([]usize),

    // Track tensors that use backward chaining (for recursive calls)
    backward_tensors: std.StringHashMapUnmanaged(void),

    // Track sparse tensors (COO format)
    sparse_tensors: std.StringHashMapUnmanaged(sparse.SparseInfo),

    // Track Tucker-decomposed tensors
    tucker_tensors: std.StringHashMapUnmanaged(tucker.TuckerInfo),

    pub fn init(allocator: std.mem.Allocator) LLVMCodegen {
        return .{
            .allocator = allocator,
            .output = .{},
            .string_arena = std.heap.ArenaAllocator.init(allocator),
            .temp_counter = 0,
            .label_counter = 0,
            .equation_counter = 0,
            .tensors = .{},
            .domains = .{},
            .tensor_max_dims = .{},
            .backward_tensors = .{},
            .sparse_tensors = .{},
            .tucker_tensors = .{},
        };
    }

    pub fn deinit(self: *LLVMCodegen) void {
        self.output.deinit(self.allocator);
        self.string_arena.deinit();
        var iter = self.tensors.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.dims);
            self.allocator.free(entry.value_ptr.strides);
        }
        self.tensors.deinit(self.allocator);
        self.domains.deinit(self.allocator);
        var max_iter = self.tensor_max_dims.iterator();
        while (max_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.tensor_max_dims.deinit(self.allocator);
        self.backward_tensors.deinit(self.allocator);
        self.sparse_tensors.deinit(self.allocator);
        self.tucker_tensors.deinit(self.allocator);
    }

    // =========================================================================
    // Core emit helpers
    // =========================================================================

    pub fn newTemp(self: *LLVMCodegen) ![]const u8 {
        const arena = self.string_arena.allocator();
        const name = try std.fmt.allocPrint(arena, "%t{d}", .{self.temp_counter});
        self.temp_counter += 1;
        return name;
    }

    pub fn newLabel(self: *LLVMCodegen) ![]const u8 {
        const arena = self.string_arena.allocator();
        const name = try std.fmt.allocPrint(arena, "L{d}", .{self.label_counter});
        self.label_counter += 1;
        return name;
    }

    pub fn emit(self: *LLVMCodegen, s: []const u8) !void {
        try self.output.appendSlice(self.allocator, s);
    }

    pub fn emitFmt(self: *LLVMCodegen, comptime fmt: []const u8, args: anytype) !void {
        try self.output.writer(self.allocator).print(fmt, args);
    }

    // =========================================================================
    // Main generation entry point
    // =========================================================================

    /// Generate complete LLVM IR module
    pub fn generate(self: *LLVMCodegen, program: *const ast.Program) ![]const u8 {
        // First pass: collect domains
        for (program.statements) |stmt| {
            if (stmt == .domain_decl) {
                const d = stmt.domain_decl;
                if (d.size) |size| {
                    try self.domains.put(self.allocator, d.name, @intCast(size));
                }
            }
        }

        // Second pass: compute max dimensions for each tensor from constant indices
        for (program.statements) |stmt| {
            if (stmt == .equation) {
                const eq = stmt.equation;
                try tensor_mod.updateTensorMaxDims(self, eq.lhs.name, eq.lhs.indices);
            }
        }

        // Module header
        try self.emit(
            \\; Tensor Logic Compiled Program
            \\; Generated by tlc (Tensor Logic Compiler)
            \\
            \\; target triple inferred by clang
            \\
            \\; External declarations
            \\declare ptr @malloc(i64)
            \\declare ptr @calloc(i64, i64)
            \\declare void @free(ptr)
            \\declare i32 @printf(ptr, ...)
            \\declare double @llvm.fabs.f64(double)
            \\declare double @llvm.sqrt.f64(double)
            \\declare double @llvm.exp.f64(double)
            \\declare double @llvm.log.f64(double)
            \\declare double @llvm.sin.f64(double)
            \\declare double @llvm.cos.f64(double)
            \\declare double @llvm.pow.f64(double, double)
            \\declare double @llvm.fma.f64(double, double, double)
            \\
            \\; File I/O
            \\declare ptr @fopen(ptr, ptr)
            \\declare i64 @fwrite(ptr, i64, i64, ptr)
            \\declare i64 @fread(ptr, i64, i64, ptr)
            \\declare i32 @fclose(ptr)
            \\
            \\; Format strings
            \\@.str.tensor_start = private constant [12 x i8] c"%s[%zu] = [\00"
            \\@.str.val = private constant [6 x i8] c"%.4g \00"
            \\@.str.end = private constant [3 x i8] c"]\0A\00"
            \\@.str.newline = private constant [2 x i8] c"\0A\00"
            \\@.str.backward_result = private constant [11 x i8] c"%s = %.6g\0A\00"
            \\
        );

        // Generate tensor name strings (once per unique tensor)
        var emitted_names = std.StringHashMapUnmanaged(void){};
        defer emitted_names.deinit(self.allocator);

        for (program.statements) |stmt| {
            switch (stmt) {
                .equation => |eq| {
                    if (!emitted_names.contains(eq.lhs.name)) {
                        try emitted_names.put(self.allocator, eq.lhs.name, {});
                        try self.emitFmt("@.name.{s} = private constant [{d} x i8] c\"{s}\\00\"\n", .{
                            eq.lhs.name,
                            eq.lhs.name.len + 1,
                            eq.lhs.name,
                        });
                    }
                },
                .backward_stmt => |b| {
                    // Pre-declare gradient tensor names
                    for (b.params) |param| {
                        const grad_name = try std.fmt.allocPrint(self.string_arena.allocator(), "dL_d{s}", .{param});
                        if (!emitted_names.contains(grad_name)) {
                            try emitted_names.put(self.allocator, grad_name, {});
                            try self.emitFmt("@.name.{s} = private constant [{d} x i8] c\"{s}\\00\"\n", .{
                                grad_name,
                                grad_name.len + 1,
                                grad_name,
                            });
                        }
                    }
                },
                else => {},
            }
        }

        try self.emit("\n");

        // Identify backward queries and track which tensors need backward functions
        for (program.statements) |stmt| {
            if (stmt == .query) {
                if (backward.isBackwardQuery(&stmt.query)) {
                    try self.backward_tensors.put(self.allocator, stmt.query.tensor.name, {});
                }
            }
        }

        // Pre-allocate tensors needed for backward chaining (so compute functions can reference them)
        // This includes both the queried tensor and all tensors referenced in its equations
        var backward_deps = std.StringHashMap(void).init(self.allocator);
        defer backward_deps.deinit();

        for (program.statements) |stmt| {
            if (stmt == .equation) {
                const eq = stmt.equation;
                if (self.backward_tensors.contains(eq.lhs.name)) {
                    // Allocate the target tensor
                    if (!self.tensors.contains(eq.lhs.name)) {
                        try tensor_mod.allocateTensorGlobal(self, eq.lhs.name, eq.lhs.indices);
                    }
                    // Collect tensor dependencies from RHS
                    try collectTensorDeps(eq.rhs, &backward_deps);
                }
            }
        }

        // Allocate all dependency tensors as globals
        for (program.statements) |stmt| {
            if (stmt == .equation) {
                const eq = stmt.equation;
                if (backward_deps.contains(eq.lhs.name) and !self.backward_tensors.contains(eq.lhs.name)) {
                    if (!self.tensors.contains(eq.lhs.name)) {
                        try tensor_mod.allocateTensorGlobal(self, eq.lhs.name, eq.lhs.indices);
                    }
                }
            }
        }

        // Generate backward chaining infrastructure (memo tables + compute functions)
        // This must come before main() since compute functions are global
        try backward.genBackwardInfra(self, program);

        // Main function
        try self.emit(
            \\define i32 @main() {
            \\entry:
            \\
        );

        // Analyze statements: separate recursive from non-recursive
        var recursive_eqs = std.ArrayListUnmanaged(usize){};
        defer recursive_eqs.deinit(self.allocator);

        for (program.statements, 0..) |stmt, idx| {
            if (stmt == .equation) {
                const eq = stmt.equation;
                if (fixpoint.isRecursive(&eq)) {
                    try recursive_eqs.append(self.allocator, idx);
                }
            }
        }

        // Generate non-recursive statements first (excluding queries)
        for (program.statements, 0..) |stmt, idx| {
            // Skip queries (they come last) and recursive equations
            if (stmt == .query) continue;

            var is_recursive = false;
            for (recursive_eqs.items) |rec_idx| {
                if (rec_idx == idx) {
                    is_recursive = true;
                    break;
                }
            }
            if (!is_recursive) {
                try self.genStatement(&stmt, program);
            }
        }

        // Generate fixpoint loop for recursive equations
        if (recursive_eqs.items.len > 0) {
            try fixpoint.genFixpointLoop(self, program, recursive_eqs.items);
        }

        // Generate queries last (after fixpoint converges)
        // Separate forward and backward queries
        for (program.statements) |stmt| {
            if (stmt == .query) {
                if (backward.isBackwardQuery(&stmt.query)) {
                    try backward.genBackwardQuery(self, &stmt.query);
                } else {
                    try self.genQuery(&stmt.query);
                }
            }
        }

        // Free tensors (skip global tensors which start with @)
        var iter = self.tensors.iterator();
        while (iter.next()) |entry| {
            // Global tensors start with '@', heap tensors start with '%'
            if (entry.value_ptr.llvm_ptr.len > 0 and entry.value_ptr.llvm_ptr[0] == '%') {
                try self.emitFmt("    call void @free(ptr {s})\n", .{entry.value_ptr.llvm_ptr});
            }
        }

        try self.emit(
            \\    ret i32 0
            \\}
            \\
        );

        return self.output.items;
    }

    // =========================================================================
    // Statement generation
    // =========================================================================

    fn genStatement(self: *LLVMCodegen, stmt: *const ast.Statement, program: *const ast.Program) !void {
        switch (stmt.*) {
            .domain_decl => {}, // Already processed
            .equation => |eq| try self.genEquation(&eq),
            .query => |q| try self.genQuery(&q),
            .sparse_decl => |s| try self.genSparseDecl(&s),
            .tucker_decl => |t| try self.genTuckerDecl(&t),
            .backward_stmt => |b| try self.genBackward(&b, program),
            .save_stmt => |s| try self.genSave(&s),
            .load_stmt => |l| try self.genLoad(&l),
            else => {},
        }
    }

    /// Generate code for a tensor equation (the core of tensor logic)
    fn genEquation(self: *LLVMCodegen, eq: *const ast.Equation) !void {
        try self.emitFmt("\n    ; Equation: {s}[...] = ...\n", .{eq.lhs.name});

        // Check if this is a softmax, layer norm, or concat operation
        if (eq.rhs.* == .nonlinearity) {
            const nl = eq.rhs.nonlinearity;
            if (nl.func == .softmax) {
                try softmax.genSoftmax(self, eq);
                return;
            }
            if (nl.func == .lnorm) {
                try layernorm.genLayerNorm(self, eq);
                return;
            }
            if (nl.func == .concat) {
                try concat.genConcat(self, eq);
                return;
            }
        }

        // Ensure LHS tensor exists
        if (!self.tensors.contains(eq.lhs.name)) {
            try tensor_mod.allocateTensor(self, eq.lhs.name, eq.lhs.indices);
        }

        // Analyze the equation to find all indices
        var all_indices = std.StringHashMapUnmanaged(usize){};
        defer all_indices.deinit(self.allocator);

        // Collect LHS indices (free indices), including normalize, primed, and div indices
        var lhs_indices = std.StringHashMapUnmanaged(void){};
        defer lhs_indices.deinit(self.allocator);
        for (eq.lhs.indices) |idx| {
            const name = switch (idx) {
                .name => |n| n,
                .normalize => |n| n,
                .primed => |n| blk: {
                    // Primed index p' - create name like "p'"
                    const primed_name = std.fmt.allocPrint(self.string_arena.allocator(), "{s}'", .{n}) catch continue;
                    break :blk primed_name;
                },
                .div => |d| d.index, // Division index X/2 uses base index X
                else => continue,
            };
            const base_name = switch (idx) {
                .primed => |n| n,
                .div => |d| d.index,
                else => name,
            };
            try lhs_indices.put(self.allocator, name, {});
            const size = self.domains.get(base_name) orelse 10;
            try all_indices.put(self.allocator, name, size);
        }

        // Collect RHS indices
        try expr_mod.collectExprIndices(self, eq.rhs, &all_indices);

        // Check if this is a simple scalar/broadcast assignment
        if (all_indices.count() == 0) {
            try einsum.genScalarAssign(self, eq);
            return;
        }

        // Build index variable list, marking contracted indices
        var index_vars = std.ArrayListUnmanaged(IndexVar){};
        defer index_vars.deinit(self.allocator);

        // Free indices first (appear on LHS)
        for (eq.lhs.indices) |idx| {
            const idx_name = switch (idx) {
                .name => |n| n,
                .primed => |n| blk: {
                    const primed_name = std.fmt.allocPrint(self.string_arena.allocator(), "{s}'", .{n}) catch continue;
                    break :blk primed_name;
                },
                .div => |d| d.index, // Division index X/2 - loop variable is X
                else => continue,
            };
            const size = all_indices.get(idx_name) orelse 10;
            try index_vars.append(self.allocator, .{
                .name = idx_name,
                .size = size,
                .llvm_var = "",
                .is_contracted = false,
            });
        }

        // Contracted indices (on RHS but not LHS)
        var iter = all_indices.iterator();
        while (iter.next()) |entry| {
            if (!lhs_indices.contains(entry.key_ptr.*)) {
                try index_vars.append(self.allocator, .{
                    .name = entry.key_ptr.*,
                    .size = entry.value_ptr.*,
                    .llvm_var = "",
                    .is_contracted = true,
                });
            }
        }

        // Generate nested loops
        try einsum.genEinsumLoops(self, eq, index_vars.items);
    }

    fn genQuery(self: *LLVMCodegen, query: *const ast.Query) !void {
        const info = self.tensors.get(query.tensor.name) orelse return;

        try self.emitFmt("\n    ; Query: {s}\n", .{query.tensor.name});

        // Print tensor name
        try self.emitFmt("    call i32 (ptr, ...) @printf(ptr @.str.tensor_start, ptr @.name.{s}, i64 {d})\n", .{ query.tensor.name, info.totalSize() });

        // Print values (limit to first 20 for readability)
        const num_print = @min(info.totalSize(), 20);
        for (0..num_print) |i| {
            const ptr = try self.newTemp();
            try self.emitFmt("    {s} = getelementptr double, ptr {s}, i64 {d}\n", .{ ptr, info.llvm_ptr, i });
            const val = try self.newTemp();
            try self.emitFmt("    {s} = load double, ptr {s}\n", .{ val, ptr });
            try self.emitFmt("    call i32 (ptr, ...) @printf(ptr @.str.val, double {s})\n", .{val});
        }

        try self.emit("    call i32 (ptr, ...) @printf(ptr @.str.end)\n");
    }

    fn genSparseDecl(self: *LLVMCodegen, decl: *const ast.SparseDecl) !void {
        // Use actual sparse COO allocation
        try self.emitFmt("\n    ; Sparse tensor: {s}\n", .{decl.name});

        // Collect dimension sizes
        var dims = std.ArrayListUnmanaged(usize){};
        defer dims.deinit(self.allocator);

        for (decl.indices) |idx| {
            const domain_name = idx.domain orelse idx.name;
            const size = self.domains.get(domain_name) orelse 10;
            try dims.append(self.allocator, size);
        }

        // Estimate initial capacity (sqrt of total for sparse)
        var total: usize = 1;
        for (dims.items) |d| total *= d;
        const capacity = @max(16, @min(1024, @as(usize, @intFromFloat(@sqrt(@as(f64, @floatFromInt(total)))))));

        // Allocate sparse structure
        const sparse_info = try sparse.genSparseAlloc(self, decl.name, dims.items.len, dims.items, capacity);
        try self.sparse_tensors.put(self.allocator, decl.name, sparse_info);

        // Also create a dense view for compatibility (so queries work)
        // This allows sparse tensors to be queried while maintaining sparse storage
        var indices = std.ArrayListUnmanaged(ast.Index){};
        defer indices.deinit(self.allocator);

        for (decl.indices) |idx| {
            const domain_name = idx.domain orelse idx.name;
            try indices.append(self.allocator, .{ .name = domain_name });
        }

        try tensor_mod.allocateTensor(self, decl.name, indices.items);
    }

    fn genTuckerDecl(self: *LLVMCodegen, decl: *const ast.TuckerDecl) !void {
        try self.emitFmt("\n    ; Tucker decomposition: {s}\n", .{decl.name});

        // Get dimensions from source tensor or infer from domains
        var dims = std.ArrayListUnmanaged(usize){};
        defer dims.deinit(self.allocator);

        // If source tensor exists, get its dimensions
        if (decl.source) |source_name| {
            if (self.tensors.get(source_name)) |source_info| {
                for (source_info.dims) |d| {
                    try dims.append(self.allocator, d);
                }
            }
        }

        // If no source or source not found, use core ranks as dimension hints
        if (dims.items.len == 0) {
            for (decl.core_ranks) |r| {
                // Use rank * 2 as default dimension (approximation)
                try dims.append(self.allocator, @as(usize, @intCast(r)) * 2);
            }
        }

        // Convert core ranks to usize
        var core_ranks = std.ArrayListUnmanaged(usize){};
        defer core_ranks.deinit(self.allocator);
        for (decl.core_ranks) |r| {
            try core_ranks.append(self.allocator, @as(usize, @intCast(r)));
        }

        // Allocate Tucker structure
        const tucker_info = try tucker.genTuckerAlloc(self, decl.name, dims.items, core_ranks.items);
        try self.tucker_tensors.put(self.allocator, decl.name, tucker_info);

        // Initialize with random values (or HOSVD from source if available)
        if (decl.source) |source_name| {
            if (self.tensors.get(source_name)) |source_info| {
                try tucker.genHOSVD(self, &tucker_info, source_info.llvm_ptr);
            } else {
                try tucker.genTuckerRandomInit(self, &tucker_info);
            }
        } else {
            try tucker.genTuckerRandomInit(self, &tucker_info);
        }
    }

    fn genSave(self: *LLVMCodegen, save: *const ast.Save) !void {
        // Save tensor to binary file: save TensorName "filename"
        const info = self.tensors.get(save.tensor_name) orelse {
            try self.emitFmt("    ; Error: tensor '{s}' not found for save\n", .{save.tensor_name});
            return;
        };

        try self.emitFmt("\n    ; Save tensor '{s}' to '{s}'\n", .{ save.tensor_name, save.path });

        // Create filename string constant
        const path_len = save.path.len + 1; // +1 for null terminator
        const path_name = try std.fmt.allocPrint(self.string_arena.allocator(), "@.save_path.{s}", .{save.tensor_name});
        try self.emitFmt("    ; Path string: {s}\n", .{save.path});

        // Open file for writing (binary mode)
        const mode = try self.newTemp();
        try self.emitFmt("    {s} = alloca [3 x i8]\n", .{mode});
        try self.emitFmt("    store [3 x i8] c\"wb\\00\", ptr {s}\n", .{mode});

        const path_ptr = try self.newTemp();
        try self.emitFmt("    {s} = alloca [{d} x i8]\n", .{ path_ptr, path_len });
        try self.emitFmt("    store [{d} x i8] c\"{s}\\00\", ptr {s}\n", .{ path_len, save.path, path_ptr });

        const file = try self.newTemp();
        try self.emitFmt("    {s} = call ptr @fopen(ptr {s}, ptr {s})\n", .{ file, path_ptr, mode });

        // Write tensor data
        const total = info.totalSize();
        const written = try self.newTemp();
        try self.emitFmt("    {s} = call i64 @fwrite(ptr {s}, i64 8, i64 {d}, ptr {s})\n", .{ written, info.llvm_ptr, total, file });

        // Close file
        const close_ret = try self.newTemp();
        try self.emitFmt("    {s} = call i32 @fclose(ptr {s})\n", .{ close_ret, file });
        _ = path_name;
    }

    fn genLoad(self: *LLVMCodegen, load: *const ast.Load) !void {
        // Load tensor from binary file: load TensorName "filename"
        const info = self.tensors.get(load.tensor_name) orelse {
            try self.emitFmt("    ; Error: tensor '{s}' not found for load\n", .{load.tensor_name});
            return;
        };

        try self.emitFmt("\n    ; Load tensor '{s}' from '{s}'\n", .{ load.tensor_name, load.path });

        const path_len = load.path.len + 1;

        // Open file for reading (binary mode)
        const mode = try self.newTemp();
        try self.emitFmt("    {s} = alloca [3 x i8]\n", .{mode});
        try self.emitFmt("    store [3 x i8] c\"rb\\00\", ptr {s}\n", .{mode});

        const path_ptr = try self.newTemp();
        try self.emitFmt("    {s} = alloca [{d} x i8]\n", .{ path_ptr, path_len });
        try self.emitFmt("    store [{d} x i8] c\"{s}\\00\", ptr {s}\n", .{ path_len, load.path, path_ptr });

        const file = try self.newTemp();
        try self.emitFmt("    {s} = call ptr @fopen(ptr {s}, ptr {s})\n", .{ file, path_ptr, mode });

        // Read tensor data
        const total = info.totalSize();
        const read_count = try self.newTemp();
        try self.emitFmt("    {s} = call i64 @fread(ptr {s}, i64 8, i64 {d}, ptr {s})\n", .{ read_count, info.llvm_ptr, total, file });

        // Close file
        const close_ret = try self.newTemp();
        try self.emitFmt("    {s} = call i32 @fclose(ptr {s})\n", .{ close_ret, file });
    }

    fn genBackward(self: *LLVMCodegen, back_stmt: *const ast.Backward, program: *const ast.Program) !void {
        try self.emitFmt("\n    ; Backward pass: {s} wrt {d} params\n", .{ back_stmt.loss, back_stmt.params.len });

        // Build computation graph and derive gradients
        var ad = autodiff.Autodiff.init(self.allocator);
        defer ad.deinit();

        try ad.buildGraph(program);
        try ad.computeGradients(back_stmt.loss, back_stmt.params);

        // Generate gradient initialization (zeros)
        // First, allocate gradient tensors
        for (back_stmt.params) |param| {
            const grad_name = try std.fmt.allocPrint(self.string_arena.allocator(), "dL_d{s}", .{param});

            // Get the original tensor info to determine shape
            if (self.tensors.get(param)) |info| {
                // Allocate gradient tensor with same shape
                try self.emitFmt("    ; Allocate gradient tensor {s}\n", .{grad_name});
                const total = info.totalSize();
                const ptr = try self.newTemp();
                try self.emitFmt("    {s} = call ptr @calloc(i64 {d}, i64 8)\n", .{ ptr, total });

                // Store tensor info
                const dims = try self.allocator.dupe(usize, info.dims);
                const strides = try self.allocator.dupe(usize, info.strides);
                try self.tensors.put(self.allocator, grad_name, .{
                    .name = grad_name,
                    .llvm_ptr = ptr,
                    .rank = info.rank,
                    .dims = dims,
                    .strides = strides,
                });
            }
        }

        // Initialize dL/dLoss = 1 (gradient of loss w.r.t. itself)
        const loss_grad = try std.fmt.allocPrint(self.string_arena.allocator(), "dL_d{s}", .{back_stmt.loss});
        if (self.tensors.get(back_stmt.loss)) |loss_info| {
            // Allocate loss gradient (scalar, so size 1)
            try self.emitFmt("    ; Initialize {s} = 1\n", .{loss_grad});
            const ptr = try self.newTemp();
            try self.emitFmt("    {s} = call ptr @calloc(i64 1, i64 8)\n", .{ptr});
            try self.emitFmt("    store double 1.0, ptr {s}\n", .{ptr});

            try self.tensors.put(self.allocator, loss_grad, .{
                .name = loss_grad,
                .llvm_ptr = ptr,
                .rank = loss_info.rank,
                .dims = try self.allocator.dupe(usize, loss_info.dims),
                .strides = try self.allocator.dupe(usize, loss_info.strides),
            });
        }

        // Generate gradient equations
        for (ad.grad_equations.items) |grad_eq| {
            try self.genGradEquation(&grad_eq);
        }
    }

    fn genGradEquation(self: *LLVMCodegen, grad_eq: *const autodiff.GradEquation) !void {
        // Delegate to autodiff module for gradient codegen
        try autodiff.genGradEquation(self, grad_eq);
    }
};

/// Collect tensor names referenced in an expression
fn collectTensorDeps(expression: *const ast.Expr, deps: *std.StringHashMap(void)) !void {
    switch (expression.*) {
        .tensor_ref => |ref| {
            try deps.put(ref.name, {});
        },
        .product => |prod| {
            for (prod.factors) |factor| {
                try collectTensorDeps(factor, deps);
            }
        },
        .binary => |bin| {
            try collectTensorDeps(bin.left, deps);
            try collectTensorDeps(bin.right, deps);
        },
        .nonlinearity => |nl| try collectTensorDeps(nl.arg, deps),
        .unary => |un| try collectTensorDeps(un.operand, deps),
        .group => |g| try collectTensorDeps(g, deps),
        else => {},
    }
}

/// Compile a program to LLVM IR
pub fn compile(allocator: std.mem.Allocator, program: *const ast.Program) ![]const u8 {
    var codegen = LLVMCodegen.init(allocator);
    defer codegen.deinit();

    const ir = try codegen.generate(program);
    return try allocator.dupe(u8, ir);
}

// ============================================================================
// Tests
// ============================================================================

test "basic codegen init" {
    const allocator = std.testing.allocator;
    var codegen = LLVMCodegen.init(allocator);
    defer codegen.deinit();

    try codegen.emit("test\n");
    try std.testing.expectEqualStrings("test\n", codegen.output.items);
}
