# Syntax Highlighter Plugin - VSCode
Plugin to highlight the syntax for NXS, NXC and NXV scripts.

Supported functionalities:
* Comment and uncomment with shortcut;
* Code folding;
* Keyword suggestion;
* Code snippets.

### Instructions

#### Build
To build the plugin:
* `npx tsc`
* `npx @vscode/vsce package` or `vsce package`.

It will create a `.vsix` file.

#### Installation
**Method 1: Using the Extensions Menu**
- Open your main Visual Studio Code window.
- Open the **Extensions view** by clicking the square blocks icon on the left sidebar (or press `Ctrl+Shift+X` on Windows / `Cmd+Shift+X` on Mac).
- At the top right of the Extensions panel, click the `...` (**Views and More Actions**) icon.
- Select **Install from VSIX...** from the dropdown menu.
- Browse to your extension folder, select the `nemantix-x.y.z.vsix` file, and click Install.

**Method 2: Using the Command Palette**
- Open VS Code.
- Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac) to open the Command Palette.
- Type **Extensions: Install from VSIX...** and press Enter.
- Locate and select your `.vsix` file.

**Method 3: Using the Terminal**

In a terminal, write:
```bash
code --install-extension nemantix-x.y.z.vsix
```