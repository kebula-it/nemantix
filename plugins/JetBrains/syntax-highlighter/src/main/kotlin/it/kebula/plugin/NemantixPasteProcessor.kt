package it.kebula.plugin

import com.intellij.codeInsight.editorActions.CopyPastePostProcessor
import com.intellij.codeInsight.editorActions.TextBlockTransferableData
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.RangeMarker
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.Ref
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiFile
import com.intellij.psi.util.PsiUtilBase
import java.awt.datatransfer.DataFlavor
import java.awt.datatransfer.Transferable

class NemantixPasteProcessor : CopyPastePostProcessor<TextBlockTransferableData>() {

    // Dummy tracking object to force IntelliJ to trigger our processor on any text paste
    class NemantixTransferableData : TextBlockTransferableData {
        companion object {
            val FLAVOR = DataFlavor(NemantixTransferableData::class.java, "NemantixPasteMarker")
        }
        override fun getFlavor(): DataFlavor = FLAVOR
        override fun getOffsetCount(): Int = 0
        override fun setOffsets(offsets: IntArray, index: Int): Int = index
        override fun getOffsets(offsets: IntArray, index: Int): Int = index
    }

    override fun collectTransferableData(
        file: PsiFile, editor: Editor, startOffsets: IntArray, endOffsets: IntArray
    ): List<TextBlockTransferableData> = emptyList()

    override fun extractTransferableData(content: Transferable): List<TextBlockTransferableData> {
        if (content.isDataFlavorSupported(DataFlavor.stringFlavor)) {
            return listOf(NemantixTransferableData())
        }
        return emptyList()
    }

    override fun processTransferableData(
        project: Project, editor: Editor, bounds: RangeMarker,
        caretOffset: Int, indented: Ref<in Boolean>, values: List<TextBlockTransferableData>
    ) {
        val file = PsiUtilBase.getPsiFileInEditor(editor, project) ?: return
        if (file.language.id != "Nemantix") return

        val document = editor.document
        val start = bounds.startOffset
        val end = bounds.endOffset
        if (start >= end) return

        val originalText = document.getText(TextRange(start, end))

        val startLine = document.getLineNumber(start)
        val lineStartOffset = document.getLineStartOffset(startLine)
        val prefix = document.getText(TextRange(lineStartOffset, start))

        // No math! Just extract the exact literal string of spaces/tabs that exists at the start.
        val baseIndentString = if (start == lineStartOffset) {
            originalText.takeWhile { it == ' ' || it == '\t' }
        } else {
            prefix.takeWhile { it == ' ' || it == '\t' }
        }

        var cleanedText = originalText

        // 1. Shield Python Blocks safely using @@@
        val pythonBlocks = mutableListOf<String>()
        cleanedText = Regex(">>>[\\s\\S]*?<<<").replace(cleanedText) { match ->
            pythonBlocks.add(match.value)
            "@@@PYTHON_BLOCK_${pythonBlocks.size - 1}@@@"
        }

        // 2. Heuristic Cleanup
        cleanedText = cleanedText.replace(Regex("(?<=\\S)\\s{3,}"), "\n")
        cleanedText = cleanedText.replace(Regex("(__[a-zA-Z0-9]*)"), "\n$1\n")
        cleanedText = cleanedText.replace(Regex("(?<!_)(deliberate\\s+)"), "\n$1")

        // 3. Process line by line
        val lines = cleanedText.split("\n").map { it.trim() }.filter { it.isNotEmpty() }
        val formattedText = StringBuilder()
        val tab = "    "
        var isFirstOutputLine = true

        var relativeIndentLevel = 0
        val containerIndentStack = mutableListOf<Int>() // Tracks visual depth of nested () and []

        fun appendFormatted(text: String, currentTotalIndent: Int, isRaw: Boolean = false) {
            if (isRaw) {
                formattedText.append(text).append("\n")
            } else {
                if (isFirstOutputLine && start > lineStartOffset) {
                    formattedText.append(text).append("\n")
                } else {
                    formattedText.append(baseIndentString)
                        .append(tab.repeat(currentTotalIndent))
                        .append(text)
                        .append("\n")
                }
            }
            isFirstOutputLine = false
        }

        val startBlockRegex = Regex("^(?:@[a-zA-Z0-9_:-]+\\s*)*(?:(?:frozen|drafted|undefined)\\s+)?(?:plan:|action\\b|body:|in:|out:|mandate:|guidelines:|deliberate\\b|toolset\\b|use:|if\\b|elif\\b|else\\b|repeat\\b|while\\b|until\\b|for\\b|>>>)")

        val continuationKeywordRegex = Regex("^(elif|else)\\b")

        // Helper function to process standard lines and boundary lines identically
        fun processLineFormatting(textLine: String) {
            val isContinuation = continuationKeywordRegex.containsMatchIn(textLine)
            if (textLine.startsWith("__") || textLine.startsWith("<<<") || isContinuation) {
                relativeIndentLevel = maxOf(0, relativeIndentLevel - 1)
            }

            // Strip strings so we don't count brackets inside text like "Ticket-[ID]"
            val codeOnly = textLine.replace(Regex("\"[^\"]*\""), "")

            val opens = codeOnly.count { it == '(' || it == '[' }
            val closes = codeOnly.count { it == ')' || it == ']' }

            // Count how many brackets close at the very beginning of the line
            val closesAtStart = codeOnly.takeWhile { it == ')' || it == ']' || it.isWhitespace() }
                .count { it == ')' || it == ']' }

            // Pull indentation back out immediately if the line starts with closing brackets
            for (i in 0 until closesAtStart) {
                if (containerIndentStack.isNotEmpty()) containerIndentStack.removeLast()
            }

            val currentContainerIndent = containerIndentStack.lastOrNull() ?: 0

            // Print the line with both structural and container indentation combined
            appendFormatted(textLine, relativeIndentLevel + currentContainerIndent)

            if (startBlockRegex.containsMatchIn(textLine)) {
                relativeIndentLevel++
            }

            // Process the net bracket change for the REST of the line
            val remainingCloses = closes - closesAtStart
            val netChangeAfterStart = opens - remainingCloses

            if (netChangeAfterStart > 0) {
                // If a line opens multiple containers, they all share the SAME visual indent increase
                val nextVisualIndent = currentContainerIndent + 1
                for (i in 0 until netChangeAfterStart) {
                    containerIndentStack.add(nextVisualIndent)
                }
            } else if (netChangeAfterStart < 0) {
                // Pop containers if there are trailing closures at the end of the line (e.g., `] )`)
                for (i in 0 until -netChangeAfterStart) {
                    if (containerIndentStack.isNotEmpty()) containerIndentStack.removeLast()
                }
            }
        }

        for (line in lines) {
            // Restore Python blocks safely
            if (line.contains("@@@PYTHON_BLOCK_")) {
                val resolvedText = line.replace(Regex("@@@PYTHON_BLOCK_(\\d+)@@@")) { match ->
                    val index = match.groupValues[1].toInt()
                    pythonBlocks[index]
                }

                val pyLines = resolvedText.split("\n")

                for ((i, pLine) in pyLines.withIndex()) {
                    if (i == 0 || i == pyLines.size - 1) {
                        // Process the bounds (>>> and <<<) as standard Nemantix lines
                        processLineFormatting(pLine.trim())
                    } else {
                        // Internal Python lines remain raw
                        appendFormatted(pLine, 0, isRaw = true)
                    }
                }
                continue
            }

            // Standard line processing
            processLineFormatting(line)
        }

        // 4. Final Polish: Keep trailing newline if it existed
        var finalText = formattedText.toString().trimEnd()
        if (originalText.endsWith("\n")) {
            finalText += "\n"
        }

        WriteCommandAction.writeCommandAction(project).run<Throwable> {
            document.replaceString(start, end, finalText)
        }
    }
}