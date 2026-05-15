package it.kebula.plugin

import com.intellij.lang.ASTNode
import com.intellij.lang.ParserDefinition
import com.intellij.lang.PsiParser
import com.intellij.lang.PsiBuilder
import com.intellij.lexer.FlexAdapter
import com.intellij.lexer.Lexer
import com.intellij.openapi.project.Project
import com.intellij.psi.FileViewProvider
import com.intellij.psi.PsiElement
import com.intellij.psi.PsiFile
import com.intellij.psi.tree.IFileElementType
import com.intellij.psi.tree.TokenSet
import com.intellij.psi.tree.IElementType
import com.intellij.extapi.psi.ASTWrapperPsiElement

class NemantixParserDefinition : ParserDefinition {

    // 1. Define the File Element Type
    companion object {
        val FILE = IFileElementType(NemantixLanguage)

        // Define which tokens are comments (Critical for Ctrl+/)
        val COMMENTS = TokenSet.create(NemantixTypes.COMMENT)

        // Define which tokens are strings
        val STRINGS = TokenSet.create(NemantixTypes.STRING)
        val PROMPT_BLOCK = IElementType("PROMPT_BLOCK", NemantixLanguage)
    }

    override fun createLexer(project: Project?): Lexer = FlexAdapter(NemantixLexer(null))

    override fun getWhitespaceTokens(): TokenSet = TokenSet.create(NemantixTypes.WHITE_SPACE)
    override fun getCommentTokens(): TokenSet = COMMENTS
    override fun getStringLiteralElements(): TokenSet = STRINGS

    override fun createParser(project: Project?): PsiParser = PsiParser { root, builder ->
        val fileMark = builder.mark()
        while (!builder.eof()) {
            if (builder.tokenType == NemantixTypes.PROMPT) {
                val promptMark = builder.mark()
                while (builder.tokenType == NemantixTypes.PROMPT && !builder.eof()) {
                    builder.advanceLexer()
                }
                // USE THE NEW BLOCK TYPE HERE:
                promptMark.done(PROMPT_BLOCK)
            } else {
                builder.advanceLexer()
            }
        }
        fileMark.done(root)
        builder.treeBuilt
    }

    override fun getFileNodeType(): IFileElementType = FILE
    override fun createFile(viewProvider: FileViewProvider): PsiFile = NemantixFile(viewProvider)

    override fun createElement(node: ASTNode?): PsiElement {
        if (node?.elementType == PROMPT_BLOCK) {
            return NemantixPromptInjectionHost(node)
        }
        return ASTWrapperPsiElement(node!!)
    }
}