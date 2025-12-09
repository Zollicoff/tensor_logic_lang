const std = @import("std");
const json = std.json;
const frontend = @import("frontend");
const Lexer = frontend.Lexer;
const Parser = frontend.Parser;
const ast = frontend.ast;

/// LSP Server for Tensor Logic
pub const LspServer = struct {
    allocator: std.mem.Allocator,
    documents: std.StringHashMap(Document),
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
            \\    "hoverProvider": true
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

        const line = position.object.get("line") orelse return self.makeResponse(id, .null);
        const char = position.object.get("character") orelse return self.makeResponse(id, .null);

        _ = line;
        _ = char;

        // Get document content
        const doc = self.documents.get(uri.string) orelse return self.makeResponse(id, .null);
        _ = doc;

        // TODO: Find token at position such as tensor names, provide type/shape info
        // For now, return a simple message
        const hover_response =
            \\{
            \\  "contents": {
            \\    "kind": "markdown",
            \\    "value": "**Tensor Logic**\n\nHover information coming soon!"
            \\  }
            \\}
        ;

        return try self.makeResponseRaw(id, hover_response);
    }

    fn publishDiagnostics(self: *LspServer, uri: []const u8, content: []const u8) !void {
        const stdout_file = std.fs.File.stdout();

        // Try to parse the document
        var lexer = Lexer.init(self.allocator, content);
        const tokens = lexer.scanTokens() catch {
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
        defer self.allocator.free(tokens);

        var parser = Parser.init(self.allocator, tokens);
        _ = parser.parse() catch {
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
