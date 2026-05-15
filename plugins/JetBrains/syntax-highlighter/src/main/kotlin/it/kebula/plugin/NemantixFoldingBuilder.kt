package it.kebula.plugin

import com.intellij.lang.ASTNode
import com.intellij.lang.folding.FoldingBuilderEx
import com.intellij.lang.folding.FoldingDescriptor
import com.intellij.openapi.editor.Document
import com.intellij.openapi.project.DumbAware
import com.intellij.psi.PsiElement
import com.intellij.psi.util.PsiTreeUtil
import com.intellij.openapi.util.TextRange
import com.intellij.psi.TokenType
import com.intellij.psi.impl.source.tree.LeafPsiElement

data class OpenBlock(val keyword: String, val startOffset: Int, var name: String = "")

class NemantixFoldingBuilder : FoldingBuilderEx(), DumbAware {

    // 1. Build the list of foldable regions
    override fun buildFoldRegions(root: PsiElement, document: Document, quick: Boolean): Array<FoldingDescriptor> {
        val descriptors = ArrayList<FoldingDescriptor>()

        // Get all leaf elements (tokens) in the file
        val allTokens = PsiTreeUtil.collectElementsOfType(root, LeafPsiElement::class.java)

        // We will store start offsets when we find a start keyword
        val openBlocks = mutableListOf<OpenBlock>()

        // foldable block names definition
        val foldableBlocks = setOf("deliberate", "plan", "action", "guidelines", "toolset", "frame")
        var expectingNameFor: OpenBlock? = null

        for (token in allTokens) {
            val type = token.elementType
            val text = token.text

            // --- Capture the Name ---
            if (expectingNameFor != null) {
                // Skip spaces between "action" and "my_action"
                if (type == TokenType.WHITE_SPACE) {
                    continue
                }

                // If the next meaningful token is a name or identifier, save it!
                if (type == NemantixTypes.IDENTIFIER) {
                    expectingNameFor.name = text
                }

                // Stop looking (we either found the name, or hit something else like a colon ':')
                expectingNameFor = null
            }

            // A. Check for START keywords
            if (type == NemantixTypes.KEYWORD) {
                if (text in foldableBlocks) {
                    val block = OpenBlock(text, token.textRange.endOffset)
                    openBlocks.add(block)

                    if (text in setOf("deliberate", "action", "guidelines", "frame", "toolset")) {
                        expectingNameFor = block
                    }
                }
            }

            // B. Check for END keywords
            if (type == NemantixTypes.KEYWORD && text.startsWith("__")) {
                val endName = text.substring(2) // remove "__" -> "deliberate"

                // Find the LAST matching start block (standard nesting logic)
                val lastIndex = openBlocks.indexOfLast { it.keyword == endName }

                if (lastIndex != -1) {
                    val block = openBlocks[lastIndex]
                    val endOffset = token.textRange.startOffset

                    // Validate valid range (must not be empty)
                    if (endOffset > block.startOffset) {
                        // Form the custom placeholder string
                        val placeholderText = if (block.name.isNotEmpty()) {
                            " ${block.name} ... "  // Result: action my_action ...
                        } else {
                            "..."                  // Result: plan...
                        }

                        // Create a custom FoldingDescriptor that uses our specific placeholder
                        descriptors.add(object : FoldingDescriptor(token.parent.node, TextRange(block.startOffset, token.textRange.endOffset)) {
                            override fun getPlaceholderText(): String = placeholderText
                        })
                    }

                    openBlocks.subList(lastIndex, openBlocks.size).clear()
                    expectingNameFor = null // Reset safety
                }
            }
        }

        return descriptors.toTypedArray()
    }

    // 2. Define what is shown when collapsed (e.g. "...")
    override fun getPlaceholderText(node: ASTNode): String {
        return "..."
    }

    // 3. Define initial state (collapsed by default? usually no)
    override fun isCollapsedByDefault(node: ASTNode): Boolean {
        return false
    }
}