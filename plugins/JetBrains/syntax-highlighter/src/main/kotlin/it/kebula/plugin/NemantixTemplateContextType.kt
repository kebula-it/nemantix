package it.kebula.plugin

import com.intellij.codeInsight.template.TemplateActionContext
import com.intellij.codeInsight.template.TemplateContextType

class NemantixTemplateContextType : TemplateContextType("NEMANTIX", "Nemantix") {
    override fun isInContext(templateActionContext: TemplateActionContext): Boolean {
        val file = templateActionContext.file

        // 1. Check if the physical file has your extension (solves Ctrl+J)
        val isNemantixFile = file.name.endsWith(".nxs") ||
                file.name.endsWith(".nxc") ||
                file.name.endsWith(".nxv")

        // 2. Check the language ID safely (solves typing autocomplete)
        val isNemantixLanguage = file.language.id.equals("Nemantix", ignoreCase = true)

        return isNemantixFile || isNemantixLanguage
    }
}