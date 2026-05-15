package it.kebula.plugin

import com.intellij.lang.Language
import com.intellij.openapi.fileTypes.LanguageFileType
import javax.swing.Icon

object NemantixLanguage : Language("Nemantix")

object NemantixFileType : LanguageFileType(NemantixLanguage) {
    override fun getName() = "Nemantix"
    override fun getDescription() = "Nemantix intentional language file"
    override fun getDefaultExtension() = "nxs"
    override fun getIcon(): Icon = NemantixIcons.FILE
}