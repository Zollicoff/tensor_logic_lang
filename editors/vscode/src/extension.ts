import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('Tensor Logic extension is now active');

    // Get the server path from settings or use default
    const config = vscode.workspace.getConfiguration('tensorlogic');
    let serverPath = config.get<string>('server.path');

    if (!serverPath) {
        // Try to find tlc-lsp in common locations
        const possiblePaths = [
            path.join(context.extensionPath, '..', '..', '..', 'zig-out', 'bin', 'tlc-lsp'),
            path.join(context.extensionPath, 'bin', 'tlc-lsp'),
            'tlc-lsp'  // Assume it's in PATH
        ];

        for (const p of possiblePaths) {
            // For now, just use the first one
            serverPath = p;
            break;
        }
    }

    if (!serverPath) {
        vscode.window.showWarningMessage(
            'Tensor Logic language server not found. Syntax highlighting will work, but advanced features require the server.'
        );
        return;
    }

    // Server options - run the Zig LSP server
    const serverOptions: ServerOptions = {
        run: {
            command: serverPath,
            transport: TransportKind.stdio
        },
        debug: {
            command: serverPath,
            transport: TransportKind.stdio
        }
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'tensorlogic' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.tl')
        },
        outputChannelName: 'Tensor Logic Language Server'
    };

    // Create the language client
    client = new LanguageClient(
        'tensorlogic',
        'Tensor Logic Language Server',
        serverOptions,
        clientOptions
    );

    // Start the client (this also starts the server)
    client.start().catch(error => {
        console.error('Failed to start Tensor Logic language server:', error);
        vscode.window.showWarningMessage(
            'Failed to start Tensor Logic language server. Syntax highlighting will work, but advanced features are unavailable.'
        );
    });

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('tensorlogic.restartServer', async () => {
            if (client) {
                await client.stop();
                await client.start();
                vscode.window.showInformationMessage('Tensor Logic language server restarted');
            }
        })
    );
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
