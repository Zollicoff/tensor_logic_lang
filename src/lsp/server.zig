const std = @import("std");
const json = std.json;
const frontend = @import("frontend");
const Lexer = frontend.Lexer;
const Parser = frontend.Parser;
const ast = frontend.ast;
const tokens = frontend.tokens;

/// Symbol information for hover/go-to-definition
const SymbolInfo = struct {
    name: []const u8,
    kind: SymbolKind,
    location: tokens.SourceLocation,
    detail: []const u8, // Type info, domain size, etc.

    const SymbolKind = enum {
        tensor,
        domain,
        sparse_relation,
    };
};

/// LSP Server for Tensor Logic
pub const LspServer = struct {
    allocator: std.mem.Allocator,
    documents: std.StringHashMap(Document),
    symbols: std.StringHashMap(std.ArrayListUnmanaged(SymbolInfo)),
    initialized: bool,
    shutdown_requested: bool,

    const Document = struct {
        uri: []const u8,
        content: []const u8,
        version: i64,
    };

    pub fn init(allocator: std.mem.Allocator) LspServer {
        return .{
            .allocator = allocator,
            .documents = std.StringHashMap(Document).init(allocator),
            .symbols = std.StringHashMap(std.ArrayListUnmanaged(SymbolInfo)).init(allocator),
            .initialized = false,
            .shutdown_requested = false,
        };
    }

    pub fn deinit(self: *LspServer) void {
        var it = self.documents.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.content);
            self.allocator.free(entry.value_ptr.uri);
        }
        self.documents.deinit();

        var sym_it = self.symbols.iterator();
        while (sym_it.next()) |entry| {
            for (entry.value_ptr.items) |sym| {
                self.allocator.free(sym.detail);
            }
            entry.value_ptr.deinit(self.allocator);
        }
        self.symbols.deinit();
    }

    /// Main loop - read JSON-RPC messages from stdin, process, respond on stdout
    pub fn run(self: *LspServer) !void {
        const stdin_file = std.fs.File.stdin();
        const stdout_file = std.fs.File.stdout();

        while (!self.shutdown_requested) {
            // Read Content-Length header
            var header_buf: [256]u8 = undefined;
            var header_len: usize = 0;
            while (header_len < header_buf.len - 1) {
                var byte_buf: [1]u8 = undefined;
                const n = stdin_file.read(&byte_buf) catch break;
                if (n == 0) return; // EOF
                if (byte_buf[0] == '\n') break;
                header_buf[header_len] = byte_buf[0];
                header_len += 1;
            }
            const header_line = header_buf[0..header_len];

            // Parse Content-Length
            const content_length = parseContentLength(header_line) orelse continue;

            // Skip empty line after headers (read until we get a blank line)
            var skip_buf: [256]u8 = undefined;
            var skip_len: usize = 0;
            while (skip_len < skip_buf.len - 1) {
                var byte_buf: [1]u8 = undefined;
                const n = stdin_file.read(&byte_buf) catch break;
                if (n == 0) break;
                if (byte_buf[0] == '\n') break;
                skip_buf[skip_len] = byte_buf[0];
                skip_len += 1;
            }

            // Read the JSON body
            const body = self.allocator.alloc(u8, content_length) catch continue;
            defer self.allocator.free(body);

            const bytes_read = stdin_file.readAll(body) catch continue;
            if (bytes_read != content_length) continue;

            // Parse and handle the message
            const response = self.handleMessage(body) catch continue;

            if (response) |resp| {
                defer self.allocator.free(resp);
                // Write response with Content-Length header
                const header = std.fmt.allocPrint(self.allocator, "Content-Length: {d}\r\n\r\n", .{resp.len}) catch continue;
                defer self.allocator.free(header);
                stdout_file.writeAll(header) catch continue;
                stdout_file.writeAll(resp) catch continue;
            }
        }
    }

    fn parseContentLength(header: []const u8) ?usize {
        const prefix = "Content-Length: ";
        const trimmed = std.mem.trim(u8, header, &[_]u8{ '\r', '\n', ' ' });
        if (std.mem.startsWith(u8, trimmed, prefix)) {
            const num_str = trimmed[prefix.len..];
            return std.fmt.parseInt(usize, num_str, 10) catch null;
        }
        return null;
    }

    fn handleMessage(self: *LspServer, body: []const u8) !?[]const u8 {
        const parsed = json.parseFromSlice(json.Value, self.allocator, body, .{}) catch return null;
        defer parsed.deinit();

        const root = parsed.value;
        const method = root.object.get("method") orelse return null;
        const method_str = method.string;

        const id = root.object.get("id");

        // Handle different methods
        if (std.mem.eql(u8, method_str, "initialize")) {
            return try self.handleInitialize(id);
        } else if (std.mem.eql(u8, method_str, "initialized")) {
            self.initialized = true;
            return null; // Notification, no response
        } else if (std.mem.eql(u8, method_str, "shutdown")) {
            self.shutdown_requested = true;
            return try self.makeResponse(id, .null);
        } else if (std.mem.eql(u8, method_str, "exit")) {
            return null;
        } else if (std.mem.eql(u8, method_str, "textDocument/didOpen")) {
            try self.handleDidOpen(root);
            return null;
        } else if (std.mem.eql(u8, method_str, "textDocument/didChange")) {
            try self.handleDidChange(root);
            return null;
        } else if (std.mem.eql(u8, method_str, "textDocument/didClose")) {
            try self.handleDidClose(root);
            return null;
        } else if (std.mem.eql(u8, method_str, "textDocument/hover")) {
            return try self.handleHover(root, id);
        } else if (std.mem.eql(u8, method_str, "textDocument/definition")) {
            return try self.handleDefinition(root, id);
        } else if (std.mem.eql(u8, method_str, "textDocument/completion")) {
            return try self.handleCompletion(root, id);
        }

        return null;
    }

    fn handleInitialize(self: *LspServer, id: ?json.Value) ![]const u8 {
        const capabilities =
            \\{
            \\  "capabilities": {
            \\    "textDocumentSync": {
            \\      "openClose": true,
            \\      "change": 1
            \\    },
            \\    "hoverProvider": true,
            \\    "definitionProvider": true,
            \\    "completionProvider": {
            \\      "triggerCharacters": ["[", "(", "=", " "]
            \\    }
            \\  },
            \\  "serverInfo": {
            \\    "name": "tensor-logic-lsp",
            \\    "version": "0.1.0"
            \\  }
            \\}
        ;
        return try self.makeResponseRaw(id, capabilities);
    }

    fn handleDidOpen(self: *LspServer, root: json.Value) !void {
        const params = root.object.get("params") orelse return;
        const text_doc = params.object.get("textDocument") orelse return;
        const uri = text_doc.object.get("uri") orelse return;
        const text = text_doc.object.get("text") orelse return;
        const version = text_doc.object.get("version") orelse return;

        const uri_copy = try self.allocator.dupe(u8, uri.string);
        const text_copy = try self.allocator.dupe(u8, text.string);

        try self.documents.put(uri_copy, .{
            .uri = uri_copy,
            .content = text_copy,
            .version = version.integer,
        });

        // Publish diagnostics
        try self.publishDiagnostics(uri.string, text.string);
    }

    fn handleDidChange(self: *LspServer, root: json.Value) !void {
        const params = root.object.get("params") orelse return;
        const text_doc = params.object.get("textDocument") orelse return;
        const uri = text_doc.object.get("uri") orelse return;
        const changes = params.object.get("contentChanges") orelse return;

        if (changes.array.items.len > 0) {
            const change = changes.array.items[0];
            const new_text = change.object.get("text") orelse return;

            if (self.documents.getPtr(uri.string)) |doc| {
                self.allocator.free(doc.content);
                doc.content = try self.allocator.dupe(u8, new_text.string);
            }

            // Publish diagnostics
            try self.publishDiagnostics(uri.string, new_text.string);
        }
    }

    fn handleDidClose(self: *LspServer, root: json.Value) !void {
        const params = root.object.get("params") orelse return;
        const text_doc = params.object.get("textDocument") orelse return;
        const uri = text_doc.object.get("uri") orelse return;

        if (self.documents.fetchRemove(uri.string)) |kv| {
            self.allocator.free(kv.key);
            self.allocator.free(kv.value.content);
            self.allocator.free(kv.value.uri);
        }
    }

    fn handleHover(self: *LspServer, root: json.Value, id: ?json.Value) ![]const u8 {
        const params = root.object.get("params") orelse return self.makeResponse(id, .null);
        const text_doc = params.object.get("textDocument") orelse return self.makeResponse(id, .null);
        const uri = text_doc.object.get("uri") orelse return self.makeResponse(id, .null);
        const position = params.object.get("position") orelse return self.makeResponse(id, .null);

        const line_num = position.object.get("line") orelse return self.makeResponse(id, .null);
        const char_num = position.object.get("character") orelse return self.makeResponse(id, .null);

        // Get document content
        const doc = self.documents.get(uri.string) orelse return self.makeResponse(id, .null);

        // Find the word at the cursor position
        const word = self.getWordAtPosition(doc.content, @intCast(line_num.integer), @intCast(char_num.integer)) orelse return self.makeResponse(id, .null);

        // Look up symbol info
        if (self.symbols.get(uri.string)) |syms| {
            for (syms.items) |sym| {
                if (std.mem.eql(u8, sym.name, word)) {
                    const kind_str = switch (sym.kind) {
                        .tensor => "Tensor",
                        .domain => "Domain",
                        .sparse_relation => "Sparse Relation",
                    };
                    const hover_response = try std.fmt.allocPrint(self.allocator,
                        \\{{
                        \\  "contents": {{
                        \\    "kind": "markdown",
                        \\    "value": "**{s}** `{s}`\n\n{s}"
                        \\  }}
                        \\}}
                    , .{ kind_str, sym.name, sym.detail });
                    defer self.allocator.free(hover_response);
                    return try self.makeResponseRaw(id, hover_response);
                }
            }
        }

        // Check if it's a keyword or nonlinearity
        if (self.getKeywordHover(word)) |info| {
            const hover_response = try std.fmt.allocPrint(self.allocator,
                \\{{
                \\  "contents": {{
                \\    "kind": "markdown",
                \\    "value": "{s}"
                \\  }}
                \\}}
            , .{info});
            defer self.allocator.free(hover_response);
            return try self.makeResponseRaw(id, hover_response);
        }

        return self.makeResponse(id, .null);
    }

    fn handleDefinition(self: *LspServer, root: json.Value, id: ?json.Value) ![]const u8 {
        const params = root.object.get("params") orelse return self.makeResponse(id, .null);
        const text_doc = params.object.get("textDocument") orelse return self.makeResponse(id, .null);
        const uri = text_doc.object.get("uri") orelse return self.makeResponse(id, .null);
        const position = params.object.get("position") orelse return self.makeResponse(id, .null);

        const line_num = position.object.get("line") orelse return self.makeResponse(id, .null);
        const char_num = position.object.get("character") orelse return self.makeResponse(id, .null);

        // Get document content
        const doc = self.documents.get(uri.string) orelse return self.makeResponse(id, .null);

        // Find the word at the cursor position
        const word = self.getWordAtPosition(doc.content, @intCast(line_num.integer), @intCast(char_num.integer)) orelse return self.makeResponse(id, .null);

        // Look up symbol location
        if (self.symbols.get(uri.string)) |syms| {
            for (syms.items) |sym| {
                if (std.mem.eql(u8, sym.name, word)) {
                    // LSP uses 0-based lines, our parser uses 1-based
                    const def_line = if (sym.location.line > 0) sym.location.line - 1 else 0;
                    const def_response = try std.fmt.allocPrint(self.allocator,
                        \\{{
                        \\  "uri": "{s}",
                        \\  "range": {{
                        \\    "start": {{ "line": {d}, "character": {d} }},
                        \\    "end": {{ "line": {d}, "character": {d} }}
                        \\  }}
                        \\}}
                    , .{ uri.string, def_line, sym.location.column, def_line, sym.location.column + sym.name.len });
                    defer self.allocator.free(def_response);
                    return try self.makeResponseRaw(id, def_response);
                }
            }
        }

        return self.makeResponse(id, .null);
    }

    fn handleCompletion(self: *LspServer, root: json.Value, id: ?json.Value) ![]const u8 {
        const params = root.object.get("params") orelse return self.makeResponse(id, .null);
        const text_doc = params.object.get("textDocument") orelse return self.makeResponse(id, .null);
        const uri = text_doc.object.get("uri") orelse return self.makeResponse(id, .null);

        var completions = std.ArrayListUnmanaged(u8){};
        defer completions.deinit(self.allocator);

        try completions.appendSlice(self.allocator, "[");
        var first = true;

        // Add keywords
        const keywords = [_][]const u8{ "domain", "sparse", "import", "export", "if", "else", "save", "load", "tucker", "backward" };
        for (keywords) |kw| {
            if (!first) try completions.appendSlice(self.allocator, ",");
            first = false;
            const item = try std.fmt.allocPrint(self.allocator,
                \\{{"label": "{s}", "kind": 14, "detail": "keyword"}}
            , .{kw});
            defer self.allocator.free(item);
            try completions.appendSlice(self.allocator, item);
        }

        // Add nonlinearities
        const nonlinearities = [_][]const u8{ "step", "relu", "sigmoid", "softmax", "tanh", "exp", "log", "abs", "sqrt", "sin", "cos", "norm", "lnorm", "concat" };
        for (nonlinearities) |nl| {
            if (!first) try completions.appendSlice(self.allocator, ",");
            first = false;
            const item = try std.fmt.allocPrint(self.allocator,
                \\{{"label": "{s}", "kind": 3, "detail": "nonlinearity"}}
            , .{nl});
            defer self.allocator.free(item);
            try completions.appendSlice(self.allocator, item);
        }

        // Add defined tensors/domains from this document
        if (self.symbols.get(uri.string)) |syms| {
            for (syms.items) |sym| {
                if (!first) try completions.appendSlice(self.allocator, ",");
                first = false;
                const kind: u8 = switch (sym.kind) {
                    .tensor => 6, // Variable
                    .domain => 22, // Struct
                    .sparse_relation => 5, // Field
                };
                const item = try std.fmt.allocPrint(self.allocator,
                    \\{{"label": "{s}", "kind": {d}, "detail": "{s}"}}
                , .{ sym.name, kind, sym.detail });
                defer self.allocator.free(item);
                try completions.appendSlice(self.allocator, item);
            }
        }

        try completions.appendSlice(self.allocator, "]");

        return try self.makeResponseRaw(id, completions.items);
    }

    /// Get word at a given line/character position
    fn getWordAtPosition(self: *LspServer, content: []const u8, line: usize, character: usize) ?[]const u8 {
        _ = self;
        var current_line: usize = 0;
        var line_start: usize = 0;

        // Find the start of the requested line
        for (content, 0..) |c, i| {
            if (current_line == line) {
                line_start = i;
                break;
            }
            if (c == '\n') {
                current_line += 1;
            }
        }

        // Find position in line
        const pos = line_start + character;
        if (pos >= content.len) return null;

        // Find word boundaries
        var start = pos;
        while (start > 0 and isIdentChar(content[start - 1])) {
            start -= 1;
        }

        var end = pos;
        while (end < content.len and isIdentChar(content[end])) {
            end += 1;
        }

        if (start == end) return null;
        return content[start..end];
    }

    fn isIdentChar(c: u8) bool {
        return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '_';
    }

    fn getKeywordHover(self: *LspServer, word: []const u8) ?[]const u8 {
        _ = self;
        const keyword_info = [_]struct { name: []const u8, info: []const u8 }{
            .{ .name = "domain", .info = "**domain** - Declare index domain size\\n\\n`domain Person: 100`" },
            .{ .name = "sparse", .info = "**sparse** - Declare sparse Boolean relation\\n\\n`sparse Parent(x: Person, y: Person)`" },
            .{ .name = "import", .info = "**import** - Import definitions from file\\n\\n`import \\\"path.tl\\\"`" },
            .{ .name = "export", .info = "**export** - Export tensor for external use\\n\\n`export TensorName`" },
            .{ .name = "save", .info = "**save** - Save tensor to file\\n\\n`save Weights \\\"weights.bin\\\"`" },
            .{ .name = "load", .info = "**load** - Load tensor from file\\n\\n`load Weights \\\"weights.bin\\\"`" },
            .{ .name = "if", .info = "**if** - Conditional expression\\n\\n`Y[i] = if X[i] > 0 then X[i] else 0`" },
            .{ .name = "step", .info = "**step** - Heaviside step function\\n\\n`H(x) = 1 if x > 0, else 0`" },
            .{ .name = "relu", .info = "**relu** - Rectified Linear Unit\\n\\n`relu(x) = max(0, x)`" },
            .{ .name = "sigmoid", .info = "**sigmoid** - Logistic sigmoid\\n\\n`sigmoid(x) = 1 / (1 + exp(-x))`" },
            .{ .name = "softmax", .info = "**softmax** - Softmax normalization\\n\\n`softmax(x_i) = exp(x_i) / sum_j(exp(x_j))`" },
            .{ .name = "tanh", .info = "**tanh** - Hyperbolic tangent\\n\\n`tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`" },
            .{ .name = "exp", .info = "**exp** - Exponential function\\n\\n`exp(x) = e^x`" },
            .{ .name = "log", .info = "**log** - Natural logarithm\\n\\n`log(x) = ln(x)`" },
            .{ .name = "lnorm", .info = "**lnorm** - Layer normalization\\n\\n`(x - mean) / std` per row" },
            .{ .name = "norm", .info = "**norm** - L2 norm\\n\\n`||x||_2 = sqrt(sum(x_i^2))`" },
            .{ .name = "concat", .info = "**concat** - Concatenation along axis" },
            .{ .name = "tucker", .info = "**tucker** - Tucker decomposition\\n\\n`tucker T(r1, r2, r3) from Source`\\n\\nDecomposes tensor into core + factor matrices for sparse→dense scaling" },
            .{ .name = "backward", .info = "**backward** - Backward chaining mode\\n\\n`backward T[0,5]?`\\n\\nQuery-driven recursive inference with memoization" },
        };

        for (keyword_info) |kw| {
            if (std.mem.eql(u8, word, kw.name)) {
                return kw.info;
            }
        }
        return null;
    }

    fn publishDiagnostics(self: *LspServer, uri: []const u8, content: []const u8) !void {
        const stdout_file = std.fs.File.stdout();

        // Try to parse the document
        var lexer = Lexer.init(self.allocator, content);
        const toks = lexer.scanTokens() catch {
            // Lexer error - report it
            const diag = try std.fmt.allocPrint(self.allocator,
                \\{{
                \\  "jsonrpc": "2.0",
                \\  "method": "textDocument/publishDiagnostics",
                \\  "params": {{
                \\    "uri": "{s}",
                \\    "diagnostics": [
                \\      {{
                \\        "range": {{ "start": {{ "line": 0, "character": 0 }}, "end": {{ "line": 0, "character": 1 }} }},
                \\        "severity": 1,
                \\        "message": "Lexer error"
                \\      }}
                \\    ]
                \\  }}
                \\}}
            , .{uri});
            defer self.allocator.free(diag);
            try self.sendNotification(stdout_file, diag);
            return;
        };
        defer self.allocator.free(toks);

        var parser = Parser.init(self.allocator, toks);
        const program = parser.parse() catch {
            // Parser error - report it
            const errors = parser.getErrors();
            var diag_items = std.ArrayListUnmanaged(u8){};
            defer diag_items.deinit(self.allocator);

            try diag_items.appendSlice(self.allocator, "[");
            for (errors, 0..) |err, i| {
                if (i > 0) try diag_items.appendSlice(self.allocator, ",");
                const item = try std.fmt.allocPrint(self.allocator,
                    \\{{
                    \\  "range": {{ "start": {{ "line": {d}, "character": {d} }}, "end": {{ "line": {d}, "character": {d} }} }},
                    \\  "severity": 1,
                    \\  "message": "{s}"
                    \\}}
                , .{ err.location.line, err.location.column, err.location.line, err.location.column + 1, err.message });
                defer self.allocator.free(item);
                try diag_items.appendSlice(self.allocator, item);
            }
            try diag_items.appendSlice(self.allocator, "]");

            const diag = try std.fmt.allocPrint(self.allocator,
                \\{{
                \\  "jsonrpc": "2.0",
                \\  "method": "textDocument/publishDiagnostics",
                \\  "params": {{
                \\    "uri": "{s}",
                \\    "diagnostics": {s}
                \\  }}
                \\}}
            , .{ uri, diag_items.items });
            defer self.allocator.free(diag);
            try self.sendNotification(stdout_file, diag);
            return;
        };

        // Extract symbols from AST
        try self.extractSymbols(uri, &program);

        // No errors - clear diagnostics
        const clear_diag = try std.fmt.allocPrint(self.allocator,
            \\{{
            \\  "jsonrpc": "2.0",
            \\  "method": "textDocument/publishDiagnostics",
            \\  "params": {{
            \\    "uri": "{s}",
            \\    "diagnostics": []
            \\  }}
            \\}}
        , .{uri});
        defer self.allocator.free(clear_diag);
        try self.sendNotification(stdout_file, clear_diag);
    }

    fn extractSymbols(self: *LspServer, uri: []const u8, program: *const ast.Program) !void {
        // Clear existing symbols for this document
        if (self.symbols.getPtr(uri)) |existing| {
            for (existing.items) |sym| {
                self.allocator.free(sym.detail);
            }
            existing.clearRetainingCapacity();
        } else {
            try self.symbols.put(uri, std.ArrayListUnmanaged(SymbolInfo){});
        }

        const syms = self.symbols.getPtr(uri).?;

        for (program.statements) |stmt| {
            switch (stmt) {
                .domain_decl => |d| {
                    const detail = if (d.size) |size|
                        try std.fmt.allocPrint(self.allocator, "size: {d}", .{size})
                    else
                        try self.allocator.dupe(u8, "size: inferred");
                    try syms.append(self.allocator, .{
                        .name = d.name,
                        .kind = .domain,
                        .location = d.location,
                        .detail = detail,
                    });
                },
                .sparse_decl => |s| {
                    var detail_buf = std.ArrayListUnmanaged(u8){};
                    defer detail_buf.deinit(self.allocator);
                    try detail_buf.appendSlice(self.allocator, "indices: ");
                    for (s.indices, 0..) |idx, i| {
                        if (i > 0) try detail_buf.appendSlice(self.allocator, ", ");
                        try detail_buf.appendSlice(self.allocator, idx.name);
                        if (idx.domain) |dom| {
                            try detail_buf.appendSlice(self.allocator, ": ");
                            try detail_buf.appendSlice(self.allocator, dom);
                        }
                    }
                    const detail = try self.allocator.dupe(u8, detail_buf.items);
                    try syms.append(self.allocator, .{
                        .name = s.name,
                        .kind = .sparse_relation,
                        .location = s.location,
                        .detail = detail,
                    });
                },
                .equation => |eq| {
                    // Check if this tensor is already defined
                    var found = false;
                    for (syms.items) |sym| {
                        if (std.mem.eql(u8, sym.name, eq.lhs.name)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        // Build detail string from indices
                        var detail_buf = std.ArrayListUnmanaged(u8){};
                        defer detail_buf.deinit(self.allocator);
                        const type_str = if (eq.lhs.is_boolean) "Boolean" else "Real";
                        try detail_buf.appendSlice(self.allocator, type_str);
                        if (eq.lhs.indices.len > 0) {
                            try detail_buf.appendSlice(self.allocator, " [");
                            for (eq.lhs.indices, 0..) |idx, i| {
                                if (i > 0) try detail_buf.appendSlice(self.allocator, ", ");
                                switch (idx) {
                                    .name => |n| try detail_buf.appendSlice(self.allocator, n),
                                    .constant => |c| {
                                        const num = try std.fmt.allocPrint(self.allocator, "{d}", .{c});
                                        defer self.allocator.free(num);
                                        try detail_buf.appendSlice(self.allocator, num);
                                    },
                                    else => try detail_buf.appendSlice(self.allocator, "..."),
                                }
                            }
                            try detail_buf.appendSlice(self.allocator, "]");
                        }
                        const detail = try self.allocator.dupe(u8, detail_buf.items);
                        try syms.append(self.allocator, .{
                            .name = eq.lhs.name,
                            .kind = .tensor,
                            .location = eq.location,
                            .detail = detail,
                        });
                    }
                },
                else => {},
            }
        }
    }

    fn sendNotification(self: *LspServer, file: std.fs.File, msg: []const u8) !void {
        const header = try std.fmt.allocPrint(self.allocator, "Content-Length: {d}\r\n\r\n", .{msg.len});
        defer self.allocator.free(header);
        try file.writeAll(header);
        try file.writeAll(msg);
    }

    fn makeResponse(self: *LspServer, id: ?json.Value, result: json.Value) ![]const u8 {
        _ = result;
        if (id) |i| {
            return try std.fmt.allocPrint(self.allocator,
                \\{{"jsonrpc": "2.0", "id": {d}, "result": null}}
            , .{i.integer});
        }
        return try self.allocator.dupe(u8, "{\"jsonrpc\": \"2.0\", \"result\": null}");
    }

    fn makeResponseRaw(self: *LspServer, id: ?json.Value, result: []const u8) ![]const u8 {
        if (id) |i| {
            return try std.fmt.allocPrint(self.allocator,
                \\{{"jsonrpc": "2.0", "id": {d}, "result": {s}}}
            , .{ i.integer, result });
        }
        return try std.fmt.allocPrint(self.allocator,
            \\{{"jsonrpc": "2.0", "result": {s}}}
        , .{result});
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var server = LspServer.init(allocator);
    defer server.deinit();

    try server.run();
}
