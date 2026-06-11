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
            originalText.takeWhile { it == ' ' || it == '\t' }.toString()
        } else {
            prefix.takeWhile { it == ' ' || it == '\t' }.toString()
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

        // This tracks only the INTERNAL nesting of the block you pasted
        var relativeIndentLevel = 0

        fun appendFormatted(text: String, currentRelativeIndent: Int, isRaw: Boolean = false) {
            if (isRaw) {
                // Raw python code prints exactly as it was copied
                formattedText.append(text).append("\n")
            } else {
                if (isFirstOutputLine && start > lineStartOffset) {
                    // Inline paste: first line just gets text (IDE prefix already handles indent)
                    formattedText.append(text).append("\n")
                } else {
                    // Whole line paste OR subsequent lines: Print the Base String + Internal Nesting
                    formattedText.append(baseIndentString)
                        .append(tab.repeat(currentRelativeIndent))
                        .append(text)
                        .append("\n")
                }
            }
            isFirstOutputLine = false
        }

        val startBlockRegex = Regex("^(?:@[a-zA-Z0-9_:-]+\\s*)*(?:(?:frozen|drafted|undefined)\\s+)?(?:plan:|action\\b|body:|in:|out:|guidelines:|deliberate\\b|toolset\\b|use:|if\\b|elif\\b|else\\b|repeat\\b|while\\b|until\\b|for\\b|>>>)")

        for (line in lines) {

            // Restore Python blocks
            if (line.contains("@@@PYTHON_BLOCK_")) {
                val indexMatch = Regex("@@@PYTHON_BLOCK_(\\d+)@@@").find(line)
                if (indexMatch != null) {
                    val index = indexMatch.groupValues[1].toInt()
                    val pyLines = pythonBlocks[index].split("\n")

                    for (pLine in pyLines) {
                        val tLine = pLine.trim()

                        if (tLine == "<<<") relativeIndentLevel = maxOf(0, relativeIndentLevel - 1)

                        if (tLine == ">>>" || tLine == "<<<") {
                            appendFormatted(tLine, relativeIndentLevel)
                        } else {
                            appendFormatted(pLine, 0, isRaw = true)
                        }

                        if (tLine == ">>>") relativeIndentLevel++
                    }
                }
                continue
            }

            // Standard line processing
            if (line.startsWith("__") || line.startsWith("<<<")) {
                relativeIndentLevel = maxOf(0, relativeIndentLevel - 1)
            }

            appendFormatted(line, relativeIndentLevel)

            if (startBlockRegex.containsMatchIn(line)) {
                relativeIndentLevel++
            }
        }

        // 4. Final Polish: Keep trailing newline if it existed
        var finalText = formattedText.toString().trimEnd()
        if (originalText.endsWith("\n")) {
            finalText += "\n"
        }

        WriteCommandAction.runWriteCommandAction(project) {
            document.replaceString(start, end, finalText)
        }
    }
}