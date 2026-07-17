package it.kebula.plugin

import com.intellij.lang.injection.InjectedLanguageManager
import com.intellij.openapi.actionSystem.DataContext
import com.intellij.openapi.editor.Caret
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.actionSystem.EditorActionHandler
import com.intellij.openapi.editor.actionSystem.EditorWriteActionHandler
import com.intellij.psi.PsiDocumentManager

class NemantixCommentHandler(private val original: EditorActionHandler) : EditorWriteActionHandler(false) {

    override fun isEnabledForCaret(editor: Editor, caret: Caret, dataContext: DataContext): Boolean = true

    override fun executeWriteAction(editor: Editor, caret: Caret?, dataContext: DataContext) {
        val project = editor.project
        if (project == null) {
            original.execute(editor, caret, dataContext)
            return
        }

        val psiFile = PsiDocumentManager.getInstance(project).getPsiFile(editor.document)
        if (psiFile?.language?.id != "Nemantix") {
            original.execute(editor, caret, dataContext)
            return
        }

        val effectiveCaret = caret ?: editor.caretModel.primaryCaret
        val offset = effectiveCaret.offset
        val injectedElement = InjectedLanguageManager.getInstance(project).findInjectedElementAt(psiFile, offset)

        if (injectedElement == null) {
            // Not inside an injection — use default behavior
            original.execute(editor, caret, dataContext)
            return
        }

        // Caret is inside an injected Python fragment: apply Nemantix-level comment to the outer host line
        val document = editor.document
        val lineNumber = document.getLineNumber(offset)
        val lineStart = document.getLineStartOffset(lineNumber)
        val lineEnd = document.getLineEndOffset(lineNumber)
        val lineText = document.charsSequence.subSequence(lineStart, lineEnd).toString()

        val commentPrefix = "# "
        val trimmed = lineText.trimStart()
        val indentLength = lineText.length - trimmed.length

        if (trimmed.startsWith(commentPrefix) || trimmed.startsWith("#")) {
            // Uncomment: remove the leading # (and the following space if present)
            val toRemove = if (trimmed.startsWith(commentPrefix)) commentPrefix.length else 1
            document.deleteString(lineStart + indentLength, lineStart + indentLength + toRemove)
        } else {
            // Comment: insert "# " before the first non-whitespace character
            document.insertString(lineStart + indentLength, commentPrefix)
        }
    }
}
