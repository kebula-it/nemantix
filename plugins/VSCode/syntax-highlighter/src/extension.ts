import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    const formatter = vscode.languages.registerDocumentRangeFormattingEditProvider('nemantix', {
        provideDocumentRangeFormattingEdits(
            document: vscode.TextDocument, 
            range: vscode.Range, 
            options: vscode.FormattingOptions
        ): vscode.TextEdit[] {
            
            const originalText = document.getText(range);

            // 1. Heuristic Cleanup: Replace sequences of 3+ spaces with a newline
            let cleanedText = originalText.replace(/\s{3,}/g, '\n');

            // 2. Fix jammed keywords
            // Removed '_' from the character class so __plan__deliberate splits correctly.
            // Changed '+' to '*' so a bare '__' is also caught and put on its own line.
            cleanedText = cleanedText.replace(/(__[a-zA-Z0-9]*)/g, '\n$1\n');
            cleanedText = cleanedText.replace(/(deliberate\s+)/g, '\n$1');
            
            // 3. Process line by line to calculate proper indentation
            const lines = cleanedText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
            let formattedText = '';
            let indentLevel = 0;
            
            const tab = options.insertSpaces ? ' '.repeat(options.tabSize) : '\t';

            for (let line of lines) {
                // A. Decrease indent BEFORE writing the line if it's an end block
                if (line.startsWith('__') || line.startsWith('<<<')) {
                    indentLevel = Math.max(0, indentLevel - 1);
                }

                // Write the line with current indentation
                formattedText += tab.repeat(indentLevel) + line + '\n';

                // B. Increase indent AFTER writing the line if it's a start block
                // - ^ forces the match to the start of the line
                // - Allows optional annotations (e.g., @breakpoint) 
                // - Allows optional qualifiers (e.g., frozen, drafted, undefined)
                // - Added if, elif, else, repeat, while, until, for
                const startBlockRegex = /^(?:@[a-zA-Z0-9_:-]+\s*)*(?:(?:frozen|drafted|undefined)\s+)?(?:plan:|action\b|body:|in:|out:|guidelines:|deliberate\b|toolset\b|use:|if\b|elif\b|else\b|repeat\b|while\b|until\b|for\b|>>>)/;
                
                if (startBlockRegex.test(line)) {
                    indentLevel++;
                }
            }

            formattedText = formattedText.trimEnd();

            return [vscode.TextEdit.replace(range, formattedText)];
        }
    });

    context.subscriptions.push(formatter);
}

export function deactivate() {}
