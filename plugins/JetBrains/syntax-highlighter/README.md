# Syntax Highlighter Plugin
Plugin to highlight the syntax for NXS, NXC and NXV scripts.

Supported functionalities:
* Comment and uncomment with shortcut;
* Code folding;
* Keyword suggestion.

### Instructions

#### Install JDK
Before running any tasks, ensure IntelliJ is using the correct Java version for Gradle:
1. Open **Settings** (`Ctrl+Alt+S` or `Cmd+,`).
2. Navigate to **Build, Execution, Deployment > Build Tools > Gradle**.
3. Ensure **Gradle JVM** is set to **JetBrains Runtime 17 (JBR 17)** or another valid JDK 17.
4. Click **OK**.

#### Build
To build the plugin:
* **Generate the Lexer**:
  1. In the **Gradle** tool window, expand **Tasks > other**.
  2. Double-click `generateLexer`.
  3. **Verification:** Check your project tree.
  You should now see the generated file at `src/main/gen/it/kebula/plugin/NemantixLexer.java`.

* **Test in Sandbox** (optional):
  1. In the Gradle tool window, expand Tasks > intellij.
  2. Double-click `runIde`.
  3. A new, temporary instance of PyCharm/IntelliJ will open.
  4. Create a new project, add a `test.nxs` file, and paste some Nemantix code to verify everything is colored correctly.
  5. Close the Sandbox when you are done.

* **Build the Plugin ZIP File**:
  1. In the **Gradle** tool window, expand **Tasks > intellij**.
  2. Double-click `buildPlugin`.
  3. Locate the file: Look in your project explorer under `build/distributions/`.
  You will see your deployable file.

#### IDE Installation
To install the plugin for a JetBrains IDE:
1. Open **Settings > Plugins**.
2. Click the **Gear icon** ⚙️ at the top right of the plugins list.
3. Select **Install Plugin from Disk....**
4. Navigate to the `build/distributions/` folder and select your `.zip` file.
5. Click **OK** and restart the IDE when prompted.
