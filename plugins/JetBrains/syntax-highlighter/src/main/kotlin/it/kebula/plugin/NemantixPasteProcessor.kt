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

        // 1. Heuristic Cleanup: Replace sequences of 3+ spaces with a newline
        var cleanedText = originalText.replace(Regex("\\s{3,}"), "\n")

        // 2. Fix jammed keywords
        cleanedText = cleanedText.replace(Regex("(__[a-zA-Z0-9]*)"), "\n$1\n")
        // Use a negative lookbehind (?<!_) so it ignores '__deliberate'
        cleanedText = cleanedText.replace(Regex("(?<!_)(deliberate\\s+)"), "\n$1")

        // 3. Process line by line to calculate proper indentation
        val lines = cleanedText.split("\n").map { it.trim() }.filter { it.isNotEmpty() }
        val formattedText = StringBuilder()
        var indentLevel = 0
        val tab = "    " // Standard 4 spaces indentation

        val startBlockRegex = Regex("^(?:@[a-zA-Z0-9_:-]+\\s*)*(?:(?:frozen|drafted|undefined)\\s+)?(?:plan:|action\\b|body:|in:|out:|guidelines:|deliberate\\b|toolset\\b|use:|if\\b|elif\\b|else\\b|repeat\\b|while\\b|until\\b|for\\b|>>>)")

        for (line in lines) {
            // A. Decrease indent BEFORE writing the line if it's an end block
            if (line.startsWith("__") || line.startsWith("<<<")) {
                indentLevel = maxOf(0, indentLevel - 1)
            }

            // Write the line with current indentation
            formattedText.append(tab.repeat(indentLevel)).append(line).append("\n")

            // B. Increase indent AFTER writing the line if it's a start block
            if (startBlockRegex.containsMatchIn(line)) {
                indentLevel++
            }
        }

        val finalText = formattedText.toString().trimEnd()

        // 4. Run text replacements inside a Write Action
        WriteCommandAction.runWriteCommandAction(project) {
            document.replaceString(start, end, finalText)
        }
    }
}