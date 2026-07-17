package it.kebula.plugin

import com.intellij.lang.annotation.AnnotationHolder
import com.intellij.lang.annotation.Annotator
import com.intellij.lang.annotation.HighlightSeverity
import com.intellij.openapi.editor.DefaultLanguageHighlighterColors
import com.intellij.psi.PsiElement
import com.intellij.psi.TokenType

class NemantixAnnotator : Annotator {
    override fun annotate(element: PsiElement, holder: AnnotationHolder) {
        val elementType = element.node.elementType

        // 1. Only process tokens that the Lexer flagged as keywords or qualifiers
        if (elementType == NemantixTypes.KEYWORD ||
            elementType == NemantixTypes.DEPRECATED_KEYWORD ||
            elementType == NemantixTypes.INPUT_QUALIFIER ||
            elementType == NemantixTypes.PLAN_QUALIFIER) {

            var parenDepth = 0
            var bracketDepth = 0
            var current = element.prevSibling
            var step = 0

            // 2. Scan backwards safely without breaking prematurely on map keys
            while (current != null && step < 500) {
                val currentType = current.node.elementType

                if (currentType == NemantixTypes.LPAREN) parenDepth++
                if (currentType == NemantixTypes.RPAREN) parenDepth--
                if (currentType == NemantixTypes.LBRACKET) bracketDepth++
                if (currentType == NemantixTypes.RBRACKET) bracketDepth--

                // Only break on unambiguous block closers (like __action or __plan)
                if ((currentType == NemantixTypes.KEYWORD || currentType == NemantixTypes.DEPRECATED_KEYWORD) && current.text.startsWith("__")) {
                    break
                }

                current = current.prevSibling
                step++ // Safety limit to prevent infinite loops on massive single-line files
            }

            // Are we inside either container?
            val isInsideContainer = parenDepth > 0 || bracketDepth > 0

            if (isInsideContainer) {
                // 3. Special handling for required, optional, and default
                if (elementType == NemantixTypes.INPUT_QUALIFIER) {

                    // Scan forward to see if the next meaningful token is a Colon
                    var next = element.nextSibling
                    while (next != null && (next.node.elementType == TokenType.WHITE_SPACE || next.node.elementType == NemantixTypes.COMMENT)) {
                        next = next.nextSibling
                    }

                    // If followed by a colon, it's being used as a Map Key -> Override color
                    if (next != null && next.node.elementType == NemantixTypes.COLON) {
                        holder.newSilentAnnotation(HighlightSeverity.INFORMATION)
                            .textAttributes(DefaultLanguageHighlighterColors.IDENTIFIER)
                            .create()
                    }

                } else {
                    // 4. For all standard keywords inside containers, override to plain text...
                    // EXCEPT for value keywords like 'none' and '_' which should stay highlighted!
                    val text = element.text
                    if (text != "none" && text != "_") {
                        holder.newSilentAnnotation(HighlightSeverity.INFORMATION)
                            .textAttributes(DefaultLanguageHighlighterColors.IDENTIFIER)
                            .create()
                    }
                }
            }
        }
    }
}