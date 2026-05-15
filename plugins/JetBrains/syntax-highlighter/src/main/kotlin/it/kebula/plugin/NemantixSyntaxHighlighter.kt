package it.kebula.plugin

import it.kebula.plugin.NemantixTypes.PLAN_QUALIFIER
import com.intellij.lexer.FlexAdapter
import com.intellij.openapi.editor.DefaultLanguageHighlighterColors
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.fileTypes.SyntaxHighlighterBase
import com.intellij.openapi.fileTypes.SyntaxHighlighterFactory
import com.intellij.openapi.project.Project
import com.intellij.psi.tree.IElementType
import com.intellij.openapi.vfs.VirtualFile

class NemantixSyntaxHighlighter : SyntaxHighlighterBase() {

    // Define Color Keys
    private val PLAN_QUALIFIER_KEY = TextAttributesKey.createTextAttributesKey("NXS_PLAN_QUALIFIER", DefaultLanguageHighlighterColors.CONSTANT)
    private val INPUT_QUALIFIER_KEY = TextAttributesKey.createTextAttributesKey("NXS_INPUT_QUALIFIER", DefaultLanguageHighlighterColors.CONSTANT)

    // We reuse standard IntelliJ keys where possible so it looks good in Dark/Light themes automatically
    private val KEYWORD_KEY = TextAttributesKey.createTextAttributesKey("NXS_KEYWORD", DefaultLanguageHighlighterColors.KEYWORD)
    private val ID_KEY = TextAttributesKey.createTextAttributesKey("NXS_ID", DefaultLanguageHighlighterColors.IDENTIFIER)
    private val NUMBER_KEY = TextAttributesKey.createTextAttributesKey("NXS_NUMBER", DefaultLanguageHighlighterColors.NUMBER)
    private val STRING_KEY = TextAttributesKey.createTextAttributesKey("NXS_STRING", DefaultLanguageHighlighterColors.STRING)
    private val COMMENT_KEY = TextAttributesKey.createTextAttributesKey("NXS_COMMENT", DefaultLanguageHighlighterColors.LINE_COMMENT)

    // Custom logic for Prompts (Mapping to Doc Comment gives a nice distinct color usually)
    private val PROMPT_KEY = TextAttributesKey.createTextAttributesKey("NXS_PROMPT",
        DefaultLanguageHighlighterColors.FUNCTION_DECLARATION)
    private val PATH_KEY = TextAttributesKey.createTextAttributesKey("NXS_PATH", DefaultLanguageHighlighterColors.LABEL)
    private val OPERATOR_KEY = TextAttributesKey.createTextAttributesKey("NXS_OPERATOR", DefaultLanguageHighlighterColors.OPERATION_SIGN)
    private val META_KEY = TextAttributesKey.createTextAttributesKey("NXS_META", DefaultLanguageHighlighterColors.METADATA)

    override fun getHighlightingLexer() = FlexAdapter(NemantixLexer(null))

    override fun getTokenHighlights(tokenType: IElementType): Array<TextAttributesKey> {
        return when (tokenType) {
            NemantixTypes.KEYWORD, NemantixTypes.BOOLEAN -> arrayOf(KEYWORD_KEY)
            NemantixTypes.IDENTIFIER -> arrayOf(ID_KEY)
            NemantixTypes.NUMBER -> arrayOf(NUMBER_KEY)
            NemantixTypes.STRING -> arrayOf(STRING_KEY)
            NemantixTypes.COMMENT -> arrayOf(COMMENT_KEY)

            NemantixTypes.PROMPT -> arrayOf(PROMPT_KEY)
            NemantixTypes.PATH -> arrayOf(PATH_KEY)
            NemantixTypes.OPERATOR -> arrayOf(OPERATOR_KEY)
            NemantixTypes.META -> arrayOf(META_KEY)

            PLAN_QUALIFIER -> arrayOf(PLAN_QUALIFIER_KEY)
            NemantixTypes.INPUT_QUALIFIER -> arrayOf(INPUT_QUALIFIER_KEY)

            // Punctuation usually stays default (black/white), but you can map it if desired
            else -> emptyArray()
        }
    }
}

class NemantixSyntaxHighlighterFactory : SyntaxHighlighterFactory() {
    override fun getSyntaxHighlighter(project: Project?, virtualFile: VirtualFile?) = NemantixSyntaxHighlighter()
}