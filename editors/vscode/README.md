# Tensor Logic VS Code Extension

Language support for Tensor Logic (`.tl` files).

## Features

- **Syntax Highlighting**: Full TextMate grammar for Tensor Logic
- **Diagnostics**: Real-time parse error reporting
- **Hover Information**: Tensor type and shape information (coming soon)

## Installation

### From Source

1. Build the language server:
   ```bash
   cd /path/to/tensor_logic_lang
   zig build
   ```

2. Install the extension dependencies:
   ```bash
   cd editors/vscode
   npm install
   npm run compile
   ```

3. Link the extension to VS Code:
   ```bash
   # On macOS/Linux
   ln -s $(pwd) ~/.vscode/extensions/tensor-logic

   # Or use VS Code's developer mode
   code --extensionDevelopmentPath=$(pwd)
   ```

4. Configure the language server path in VS Code settings:
   ```json
   {
     "tensorlogic.server.path": "/path/to/tensor_logic_lang/zig-out/bin/tlc-lsp"
   }
   ```

## Syntax Highlighting

The extension provides highlighting for:
- **Keywords**: `domain`, `sparse`, `import`, `export`, `if`, `else`, `save`, `load`, `backward`, `tucker`
- **Nonlinearities**: `step`, `relu`, `sigmoid`, `softmax`, `tanh`, `exp`, `log`, `lnorm`, etc.
- **Operators**: `=`, `+=`, `*=`, `max=`, `min=`, `avg=`
- **Tensors**: Capitalized names like `X`, `Weight`, `Output`
- **Indices**: Lowercase names like `i`, `j`, `k`
- **Special indices**: Virtual (`*t`), primed (`i'`), normalize (`i.`)

## Language Server Features

The `tlc-lsp` language server provides:
- **Parse error diagnostics** - Real-time error reporting as you type
- **Hover information** - Hover over tensors, domains, and keywords to see type info and documentation
- **Go to definition** - Jump to tensor/domain definitions with F12 or Ctrl+Click
- **Autocomplete** - Suggestions for keywords, nonlinearities, and defined tensors

## Development

```bash
# Watch mode for TypeScript
npm run watch

# Build for production
npm run compile
```
