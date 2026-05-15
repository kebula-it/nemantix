package it.kebula.plugin

import com.intellij.extapi.psi.PsiFileBase
import com.intellij.psi.FileViewProvider
import com.intellij.openapi.fileTypes.FileType

class NemantixFile(viewProvider: FileViewProvider) : PsiFileBase(viewProvider, NemantixLanguage) {
    override fun getFileType(): FileType = NemantixFileType
    override fun toString(): String = "Nemantix File"
}