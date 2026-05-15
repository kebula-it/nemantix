package it.kebula.plugin

import com.intellij.lang.Commenter

class NemantixCommenter : Commenter {
    // 1. Line Comments (e.g., # comment)
    override fun getLineCommentPrefix(): String = "# "

    // 2. Block Comments (e.g., /* comment */)
    // Your grammar doesn't seem to have block comments, so we return null/null
    override fun getBlockCommentPrefix(): String? = null
    override fun getBlockCommentSuffix(): String? = null

    // 3. Commented Block (useful if you select a block and press Ctrl+Shift+/)
    override fun getCommentedBlockCommentPrefix(): String? = null
    override fun getCommentedBlockCommentSuffix(): String? = null
}