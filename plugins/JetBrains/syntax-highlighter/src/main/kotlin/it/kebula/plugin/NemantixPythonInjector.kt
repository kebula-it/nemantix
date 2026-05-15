package it.kebula.plugin

import com.intellij.lang.Language
import com.intellij.lang.injection.MultiHostInjector
import com.intellij.lang.injection.MultiHostRegistrar
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiElement

class NemantixPythonInjector : MultiHostInjector {

    override fun elementsToInjectIn(): MutableList<out Class<out PsiElement>> {
        // We only care about examining our new Host elements
        return mutableListOf(NemantixPromptInjectionHost::class.java)
    }

    override fun getLanguagesToInject(registrar: MultiHostRegistrar, context: PsiElement) {
        val text = context.text
        val startIndex = text.indexOf(">>>")
        val endIndex = text.lastIndexOf("<<<")

        // Safely check if both markers exist
        if (startIndex == -1 || endIndex == -1 || endIndex <= startIndex) return

        // 1. Traverse backwards through the flat token list
        var sibling = context.prevSibling
        var isInsideToolset = false

        while (sibling != null) {
            val siblingText = sibling.text.trim()

            if (siblingText == "toolset") {
                isInsideToolset = true
                break
            }

            if (siblingText.startsWith("__") || siblingText == "action" || siblingText == "deliberate" || siblingText == "frame" || siblingText == "plan") {
                break
            }
            sibling = sibling.prevSibling
        }

        // 2. If we are inside a toolset, inject Python!
        if (isInsideToolset) {
            val python = Language.findLanguageByID("Python")

            // Critical check: Does the testing Sandbox actually have Python installed?
            if (python == null) {
                println("NEMANTIX DEBUG: Python language not found in this IDE Sandbox!")
                return
            }

            // Safely calculate the inner content range (between >>> and <<<)
            val contentStart = startIndex + 3
            val contentEnd = endIndex

            if (contentEnd > contentStart) {
                val contentRange = TextRange(contentStart, contentEnd)
                registrar.startInjecting(python)
                registrar.addPlace(null, null, context as NemantixPromptInjectionHost, contentRange)
                registrar.doneInjecting()
            }
        }
    }
}