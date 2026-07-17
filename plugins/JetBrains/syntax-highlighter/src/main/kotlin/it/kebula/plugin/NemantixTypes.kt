package it.kebula.plugin

import com.intellij.psi.tree.IElementType
import com.intellij.psi.TokenType

class NemantixTokenType(debugName: String) : IElementType(debugName, NemantixLanguage)

object NemantixTypes {
    // Basic types
    @JvmField
    val KEYWORD = NemantixTokenType("KEYWORD")
    @JvmField
    val IDENTIFIER = NemantixTokenType("IDENTIFIER")
    @JvmField
    val STRING = NemantixTokenType("STRING")
    @JvmField
    val NUMBER = NemantixTokenType("NUMBER")
    @JvmField
    public val COMMENT = NemantixTokenType("COMMENT")

    @JvmField
    val DEPRECATED_KEYWORD = NemantixTokenType("DEPRECATED_KEYWORD")

    // do and plan qualifiers
    @JvmField
    public val PLAN_QUALIFIER = NemantixTokenType("PLAN_QUALIFIER")
    @JvmField
    public val INPUT_QUALIFIER = NemantixTokenType("INPUT_QUALIFIER")

    // NXS Specifics
    @JvmField
    val PROMPT = NemantixTokenType("PROMPT")           // For >>>...<<< and >>...
    @JvmField
    val PATH = NemantixTokenType("PATH")               // For .nxs paths
    @JvmField
    public val OPERATOR = NemantixTokenType("OPERATOR")       // +, -, ->, ==, etc.
    @JvmField
    val BOOLEAN = NemantixTokenType("BOOLEAN")         // true, false

    // Braces & punctuation
    @JvmField
    val LBRACE = NemantixTokenType("LBRACE")  // {
    @JvmField
    val RBRACE = NemantixTokenType("RBRACE")  // }
    @JvmField
    val LBRACKET = NemantixTokenType("LBRACKET") // [
    @JvmField
    val RBRACKET = NemantixTokenType("RBRACKET") // ]
    @JvmField
    public val LPAREN = NemantixTokenType("LPAREN")  // (
    @JvmField
    val RPAREN = NemantixTokenType("RPAREN")  // )
    @JvmField
    val COMMA = NemantixTokenType("COMMA")    // ,
    @JvmField
    val COLON = NemantixTokenType("COLON")    // :
    @JvmField
    val DOT = NemantixTokenType("DOT")        // .
    @JvmField
    val META = NemantixTokenType("META")      // @

    val BAD_CHARACTER = TokenType.BAD_CHARACTER
    val WHITE_SPACE = TokenType.WHITE_SPACE
}