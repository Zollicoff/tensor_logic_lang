// GPU Backend for Tensor Logic
//
// Generates GPU compute kernels for tensor operations.
// Supports multiple backends through abstraction layer.
//
// Key insight from paper: Tensor logic's foundation on tensor algebra
// makes it "natively GPU-accelerable" - all operations are dense tensor
// operations (after Tucker decomposition of sparse data).
//
// Backends:
// - CUDA (NVIDIA GPUs)
// - Metal (Apple Silicon)
// - OpenCL (cross-platform)
//
// Operations mapped to GPU:
// - Einsum contraction → matmul / tensor contraction kernels
// - Nonlinearities → elementwise kernels
// - Reductions → parallel reduce kernels
// - Softmax → reduce + elementwise

const std = @import("std");
const ast = @import("../frontend/ast.zig");
const types = @import("types.zig");

/// GPU Backend type
pub const Backend = enum {
    cuda,
    metal,
    opencl,
    cpu, // Fallback
};

/// GPU Kernel info
pub const KernelInfo = struct {
    name: []const u8,
    backend: Backend,
    source: []const u8,
    num_threads: usize,
    shared_mem: usize,
};

/// GPU Buffer info
pub const BufferInfo = struct {
    name: []const u8,
    size: usize,
    host_ptr: ?[]const u8,
    device_ptr: ?[]const u8,
};

/// Codegen context (forward declaration)
pub const CodegenContext = @import("llvm.zig").LLVMCodegen;

/// Check if GPU is available for given backend
pub fn isGPUAvailable(backend: Backend) bool {
    // TODO: Runtime detection
    return switch (backend) {
        .cuda => false, // Would check for CUDA runtime
        .metal => @import("builtin").os.tag == .macos,
        .opencl => false, // Would check for OpenCL runtime
        .cpu => true,
    };
}

/// Generate CUDA kernel for matrix multiplication
/// C[i,j] = Σ_k A[i,k] * B[k,j]
pub fn genCudaMatmul(
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
) ![]const u8 {
    _ = m;
    _ = n;
    _ = k;
    // Return CUDA kernel source
    return std.fmt.allocPrint(std.heap.page_allocator,
        \\// CUDA kernel: {s}
        \\extern "C" __global__ void {s}(
        \\    const double* A, const double* B, double* C,
        \\    int M, int N, int K
        \\) {{
        \\    int row = blockIdx.y * blockDim.y + threadIdx.y;
        \\    int col = blockIdx.x * blockDim.x + threadIdx.x;
        \\
        \\    if (row < M && col < N) {{
        \\        double sum = 0.0;
        \\        for (int i = 0; i < K; i++) {{
        \\            sum += A[row * K + i] * B[i * N + col];
        \\        }}
        \\        C[row * N + col] = sum;
        \\    }}
        \\}}
    , .{ name, name }) catch return "// CUDA kernel generation failed";
}

/// Generate Metal shader for matrix multiplication
pub fn genMetalMatmul(
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
) ![]const u8 {
    _ = m;
    _ = n;
    _ = k;
    return std.fmt.allocPrint(std.heap.page_allocator,
        \\// Metal shader: {s}
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\kernel void {s}(
        \\    device const float* A [[buffer(0)]],
        \\    device const float* B [[buffer(1)]],
        \\    device float* C [[buffer(2)]],
        \\    constant uint& M [[buffer(3)]],
        \\    constant uint& N [[buffer(4)]],
        \\    constant uint& K [[buffer(5)]],
        \\    uint2 gid [[thread_position_in_grid]]
        \\) {{
        \\    if (gid.y >= M || gid.x >= N) return;
        \\
        \\    float sum = 0.0f;
        \\    for (uint i = 0; i < K; i++) {{
        \\        sum += A[gid.y * K + i] * B[i * N + gid.x];
        \\    }}
        \\    C[gid.y * N + gid.x] = sum;
        \\}}
    , .{ name, name }) catch return "// Metal shader generation failed";
}

/// Generate CUDA kernel for elementwise ReLU
pub fn genCudaRelu(name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(std.heap.page_allocator,
        \\// CUDA kernel: {s}
        \\extern "C" __global__ void {s}(
        \\    const double* input, double* output, int n
        \\) {{
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) {{
        \\        output[idx] = input[idx] > 0.0 ? input[idx] : 0.0;
        \\    }}
        \\}}
    , .{ name, name }) catch return "// CUDA kernel generation failed";
}

/// Generate CUDA kernel for elementwise sigmoid
pub fn genCudaSigmoid(name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(std.heap.page_allocator,
        \\// CUDA kernel: {s}
        \\extern "C" __global__ void {s}(
        \\    const double* input, double* output, int n
        \\) {{
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) {{
        \\        output[idx] = 1.0 / (1.0 + exp(-input[idx]));
        \\    }}
        \\}}
    , .{ name, name }) catch return "// CUDA kernel generation failed";
}

/// Generate CUDA kernel for temperature sigmoid (embedding space reasoning)
pub fn genCudaTempSigmoid(name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(std.heap.page_allocator,
        \\// CUDA kernel: {s} (temperature sigmoid)
        \\extern "C" __global__ void {s}(
        \\    const double* input, double* output, int n, double temp
        \\) {{
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) {{
        \\        output[idx] = 1.0 / (1.0 + exp(-input[idx] / temp));
        \\    }}
        \\}}
    , .{ name, name }) catch return "// CUDA kernel generation failed";
}

/// Generate Metal shader for parallel reduction (sum)
pub fn genMetalReduce(name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(std.heap.page_allocator,
        \\// Metal shader: {s} (parallel reduction)
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\kernel void {s}(
        \\    device const float* input [[buffer(0)]],
        \\    device float* output [[buffer(1)]],
        \\    constant uint& n [[buffer(2)]],
        \\    threadgroup float* shared [[threadgroup(0)]],
        \\    uint tid [[thread_index_in_threadgroup]],
        \\    uint gid [[thread_position_in_grid]],
        \\    uint blockDim [[threads_per_threadgroup]]
        \\) {{
        \\    shared[tid] = (gid < n) ? input[gid] : 0.0f;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\
        \\    for (uint s = blockDim / 2; s > 0; s >>= 1) {{
        \\        if (tid < s) {{
        \\            shared[tid] += shared[tid + s];
        \\        }}
        \\        threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    }}
        \\
        \\    if (tid == 0) {{
        \\        output[gid / blockDim] = shared[0];
        \\    }}
        \\}}
    , .{ name, name }) catch return "// Metal shader generation failed";
}

/// Emit GPU initialization code for LLVM
pub fn emitGPUInit(ctx: *CodegenContext, backend: Backend) !void {
    switch (backend) {
        .cuda => {
            try ctx.emit("\n    ; CUDA initialization\n");
            try ctx.emit("    ; TODO: cuInit, cuDeviceGet, cuCtxCreate\n");
        },
        .metal => {
            try ctx.emit("\n    ; Metal initialization\n");
            try ctx.emit("    ; TODO: MTLCreateSystemDefaultDevice\n");
        },
        .opencl => {
            try ctx.emit("\n    ; OpenCL initialization\n");
            try ctx.emit("    ; TODO: clGetPlatformIDs, clGetDeviceIDs, clCreateContext\n");
        },
        .cpu => {
            try ctx.emit("\n    ; CPU fallback (no GPU init needed)\n");
        },
    }
}

/// Emit GPU buffer allocation
pub fn emitBufferAlloc(ctx: *CodegenContext, name: []const u8, size: usize, backend: Backend) !void {
    try ctx.emitFmt("\n    ; GPU buffer '{s}' ({d} bytes)\n", .{ name, size * 8 });

    switch (backend) {
        .cuda => {
            try ctx.emitFmt("    ; TODO: cuMemAlloc for {s}\n", .{name});
        },
        .metal => {
            try ctx.emitFmt("    ; TODO: newBufferWithLength for {s}\n", .{name});
        },
        .opencl => {
            try ctx.emitFmt("    ; TODO: clCreateBuffer for {s}\n", .{name});
        },
        .cpu => {
            // Already have CPU allocation
        },
    }
}

/// Emit GPU kernel launch
pub fn emitKernelLaunch(
    ctx: *CodegenContext,
    kernel_name: []const u8,
    grid_dim: [3]usize,
    block_dim: [3]usize,
    backend: Backend,
) !void {
    try ctx.emitFmt("\n    ; Launch kernel '{s}'\n", .{kernel_name});

    switch (backend) {
        .cuda => {
            try ctx.emitFmt("    ; cuLaunchKernel({s}, {d}, {d}, {d}, {d}, {d}, {d})\n", .{
                kernel_name,
                grid_dim[0],  grid_dim[1],  grid_dim[2],
                block_dim[0], block_dim[1], block_dim[2],
            });
        },
        .metal => {
            try ctx.emitFmt("    ; dispatchThreadgroups({s})\n", .{kernel_name});
        },
        .opencl => {
            try ctx.emitFmt("    ; clEnqueueNDRangeKernel({s})\n", .{kernel_name});
        },
        .cpu => {
            try ctx.emit("    ; CPU execution (no kernel launch)\n");
        },
    }
}

/// Emit GPU synchronization
pub fn emitSync(ctx: *CodegenContext, backend: Backend) !void {
    switch (backend) {
        .cuda => try ctx.emit("    ; cuCtxSynchronize\n"),
        .metal => try ctx.emit("    ; waitUntilCompleted\n"),
        .opencl => try ctx.emit("    ; clFinish\n"),
        .cpu => {},
    }
}

// =============================================================================
// FFI Declarations for GPU Runtime
// =============================================================================

/// Emit CUDA FFI declarations
pub fn emitCudaFFI(ctx: *CodegenContext) !void {
    try ctx.emit(
        \\
        \\; CUDA Driver API declarations
        \\declare i32 @cuInit(i32)
        \\declare i32 @cuDeviceGet(ptr, i32)
        \\declare i32 @cuCtxCreate(ptr, i32, i32)
        \\declare i32 @cuModuleLoadData(ptr, ptr)
        \\declare i32 @cuModuleGetFunction(ptr, ptr, ptr)
        \\declare i32 @cuMemAlloc(ptr, i64)
        \\declare i32 @cuMemFree(ptr)
        \\declare i32 @cuMemcpyHtoD(ptr, ptr, i64)
        \\declare i32 @cuMemcpyDtoH(ptr, ptr, i64)
        \\declare i32 @cuLaunchKernel(ptr, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr)
        \\declare i32 @cuCtxSynchronize()
        \\
    );
}

/// Emit Metal FFI declarations (via Objective-C runtime)
pub fn emitMetalFFI(ctx: *CodegenContext) !void {
    try ctx.emit(
        \\
        \\; Metal API declarations (Objective-C bridge)
        \\declare ptr @MTLCreateSystemDefaultDevice()
        \\declare ptr @objc_msgSend(ptr, ptr, ...)
        \\declare ptr @sel_registerName(ptr)
        \\
    );
}

// =============================================================================
// Kernel Registry - maps tensor operations to GPU kernels
// =============================================================================

/// Operation types that can be GPU-accelerated
pub const GPUOp = enum {
    matmul,
    elementwise_add,
    elementwise_mul,
    relu,
    sigmoid,
    temp_sigmoid,
    tanh,
    softmax,
    reduce_sum,
    reduce_max,
    reduce_min,
    layernorm,
};

/// Get optimal block size for an operation
pub fn getBlockSize(op: GPUOp) [3]usize {
    return switch (op) {
        .matmul => .{ 16, 16, 1 }, // 2D tile for matmul
        .softmax => .{ 256, 1, 1 }, // 1D for reduction-heavy
        .reduce_sum, .reduce_max, .reduce_min => .{ 256, 1, 1 },
        .layernorm => .{ 256, 1, 1 },
        else => .{ 256, 1, 1 }, // Default 1D for elementwise
    };
}

/// Calculate grid dimensions for given tensor size and block size
pub fn calcGridDim(tensor_size: [3]usize, block_size: [3]usize) [3]usize {
    return .{
        (tensor_size[0] + block_size[0] - 1) / block_size[0],
        (tensor_size[1] + block_size[1] - 1) / block_size[1],
        (tensor_size[2] + block_size[2] - 1) / block_size[2],
    };
}

// =============================================================================
// GPU Tensor Operations - Full LLVM IR generation
// =============================================================================

/// Generate GPU-accelerated einsum contraction
/// Detects if operation is matmul-like and dispatches to GPU
pub fn genGPUEinsum(
    ctx: *CodegenContext,
    result_name: []const u8,
    lhs_name: []const u8,
    rhs_name: []const u8,
    dims: struct { m: usize, n: usize, k: usize },
    backend: Backend,
) !void {
    const block_size = getBlockSize(.matmul);
    const grid_dim = calcGridDim(.{ dims.m, dims.n, 1 }, block_size);

    try ctx.emitFmt(
        \\
        \\    ; GPU matmul: {s} = {s} @ {s}
        \\    ; Dimensions: M={d}, N={d}, K={d}
        \\
    , .{ result_name, lhs_name, rhs_name, dims.m, dims.n, dims.k });

    switch (backend) {
        .cuda => {
            try ctx.emitFmt(
                \\    ; CUDA dispatch: grid({d},{d},{d}), block({d},{d},{d})
                \\    %{s}_gpu_result = call i32 @cuLaunchKernel(
                \\        ptr %matmul_kernel,
                \\        i32 {d}, i32 {d}, i32 {d},
                \\        i32 {d}, i32 {d}, i32 {d},
                \\        i32 0, ptr null, ptr %kernel_args, ptr null)
                \\    call i32 @cuCtxSynchronize()
                \\
            , .{
                grid_dim[0],  grid_dim[1],  grid_dim[2],
                block_size[0], block_size[1], block_size[2],
                result_name,
                grid_dim[0],  grid_dim[1],  grid_dim[2],
                block_size[0], block_size[1], block_size[2],
            });
        },
        .metal => {
            try ctx.emitFmt(
                \\    ; Metal dispatch: threadgroups({d},{d},{d})
                \\    ; TODO: Objective-C bridge for Metal command buffer
                \\
            , .{ grid_dim[0], grid_dim[1], grid_dim[2] });
        },
        else => {
            try ctx.emit("    ; CPU fallback for einsum\n");
        },
    }
}

/// Generate GPU-accelerated elementwise operation
pub fn genGPUElementwise(
    ctx: *CodegenContext,
    op: GPUOp,
    result_name: []const u8,
    input_name: []const u8,
    size: usize,
    backend: Backend,
) !void {
    const block_size: usize = 256;
    const grid_size = (size + block_size - 1) / block_size;

    const op_name = switch (op) {
        .relu => "relu",
        .sigmoid => "sigmoid",
        .temp_sigmoid => "temp_sigmoid",
        .tanh => "tanh",
        else => "elementwise",
    };

    try ctx.emitFmt(
        \\
        \\    ; GPU {s}: {s} = {s}({s})
        \\    ; Size: {d}, Grid: {d}, Block: {d}
        \\
    , .{ op_name, result_name, op_name, input_name, size, grid_size, block_size });

    switch (backend) {
        .cuda => {
            try ctx.emitFmt(
                \\    %{s}_gpu = call i32 @cuLaunchKernel(
                \\        ptr %{s}_kernel,
                \\        i32 {d}, i32 1, i32 1,
                \\        i32 {d}, i32 1, i32 1,
                \\        i32 0, ptr null, ptr %{s}_args, ptr null)
                \\    call i32 @cuCtxSynchronize()
                \\
            , .{ result_name, op_name, grid_size, block_size, op_name });
        },
        .metal => {
            try ctx.emit("    ; Metal dispatch for elementwise\n");
        },
        else => {},
    }
}

/// Generate GPU-accelerated softmax
pub fn genGPUSoftmax(
    ctx: *CodegenContext,
    result_name: []const u8,
    input_name: []const u8,
    size: usize,
    backend: Backend,
) !void {
    try ctx.emitFmt(
        \\
        \\    ; GPU softmax: {s} = softmax({s})
        \\    ; Two-pass: (1) reduce max, (2) exp and normalize
        \\
    , .{ result_name, input_name });

    switch (backend) {
        .cuda => {
            // Pass 1: Find max for numerical stability
            try ctx.emitFmt(
                \\    ; Pass 1: reduce_max for stability
                \\    %{s}_max = call i32 @cuLaunchKernel(ptr %reduce_max_kernel, ...)
                \\    call i32 @cuCtxSynchronize()
                \\    ; Pass 2: exp(x - max) / sum
                \\    %{s}_result = call i32 @cuLaunchKernel(ptr %softmax_kernel, ...)
                \\    call i32 @cuCtxSynchronize()
                \\
            , .{ result_name, result_name });
        },
        else => {
            try ctx.emitFmt("    ; CPU softmax for {s} (size {d})\n", .{ input_name, size });
        },
    }
}

// =============================================================================
// GPU Memory Management
// =============================================================================

/// Generate device memory allocation
pub fn genDeviceAlloc(
    ctx: *CodegenContext,
    name: []const u8,
    size_bytes: usize,
    backend: Backend,
) !void {
    try ctx.emitFmt(
        \\
        \\    ; Allocate GPU memory: {s} ({d} bytes)
        \\
    , .{ name, size_bytes });

    switch (backend) {
        .cuda => {
            try ctx.emitFmt(
                \\    %{s}_dptr = alloca ptr
                \\    %{s}_alloc = call i32 @cuMemAlloc(ptr %{s}_dptr, i64 {d})
                \\
            , .{ name, name, name, size_bytes });
        },
        .metal => {
            try ctx.emitFmt(
                \\    ; Metal: newBufferWithLength:{d} options:MTLResourceStorageModeShared
                \\
            , .{size_bytes});
        },
        else => {},
    }
}

/// Generate host-to-device copy
pub fn genH2DCopy(
    ctx: *CodegenContext,
    device_name: []const u8,
    host_name: []const u8,
    size_bytes: usize,
    backend: Backend,
) !void {
    switch (backend) {
        .cuda => {
            try ctx.emitFmt(
                \\    %{s}_h2d = call i32 @cuMemcpyHtoD(ptr %{s}_dptr, ptr %{s}, i64 {d})
                \\
            , .{ device_name, device_name, host_name, size_bytes });
        },
        .metal => {
            try ctx.emit("    ; Metal: contents memcpy\n");
        },
        else => {},
    }
}

/// Generate device-to-host copy
pub fn genD2HCopy(
    ctx: *CodegenContext,
    host_name: []const u8,
    device_name: []const u8,
    size_bytes: usize,
    backend: Backend,
) !void {
    switch (backend) {
        .cuda => {
            try ctx.emitFmt(
                \\    %{s}_d2h = call i32 @cuMemcpyDtoH(ptr %{s}, ptr %{s}_dptr, i64 {d})
                \\
            , .{ host_name, host_name, device_name, size_bytes });
        },
        .metal => {
            try ctx.emit("    ; Metal: contents memcpy back\n");
        },
        else => {},
    }
}

/// Generate device memory free
pub fn genDeviceFree(
    ctx: *CodegenContext,
    name: []const u8,
    backend: Backend,
) !void {
    switch (backend) {
        .cuda => {
            try ctx.emitFmt(
                \\    %{s}_free = call i32 @cuMemFree(ptr %{s}_dptr)
                \\
            , .{ name, name });
        },
        else => {},
    }
}
