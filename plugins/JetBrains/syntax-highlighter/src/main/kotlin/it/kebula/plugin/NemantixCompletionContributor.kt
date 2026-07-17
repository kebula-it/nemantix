package it.kebula.plugin

import com.intellij.codeInsight.completion.*
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.patterns.PlatformPatterns
import com.intellij.util.ProcessingContext

class NemantixCompletionContributor : CompletionContributor() {

    // 1. Define your keyword list
    private val KEYWORDS = listOf(
        // Structure
        "require", "deliberate", "toolset", "frame", "slot", "plan",
        "action", "tool", "mandate", "body", "in", "out",

        // Flow Control
        "if", "elif", "else",
        "repeat", "while", "until", "each", "times",
        "do", "return", "break", "continue",

        // Modifiers
        "when", "from", "use", "as", "where", "max",
        "using", "producing", "required", "optional", "default",

        // Values / States
        "true", "false", "none",
        "undefined", "drafted", "frozen"
    )

    private val DEPRECATED_KEYWORDS = listOf("guidelines")

    init {
        // 2. Extend the completion logic
        extend(
            CompletionType.BASIC,
            PlatformPatterns.psiElement().withLanguage(NemantixLanguage), // Trigger anywhere in a .nxs file
            object : CompletionProvider<CompletionParameters>() {
                override fun addCompletions(
                    parameters: CompletionParameters,
                    context: ProcessingContext,
                    result: CompletionResultSet
                ) {
                    for (keyword in KEYWORDS) {
                        result.addElement(LookupElementBuilder.create(keyword))
                    }
                    for (keyword in DEPRECATED_KEYWORDS) {
                        result.addElement(
                            LookupElementBuilder.create(keyword)
                                .withTailText(" deprecated → use mandate", true)
                                .withStrikeoutness(true)
                        )
                    }
                }
            }
        )
    }
}