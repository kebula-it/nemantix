plugins {
//    kotlin("jvm") version "2.3.0"
    id("org.jetbrains.kotlin.jvm") version "1.9.22"
    id("org.jetbrains.intellij") version "1.17.2"
    id("org.jetbrains.grammarkit") version "2022.3.2.2"
    id("org.jetbrains.changelog") version "2.2.1"
}

group = "it.kebula"
version = "1.1.3"

repositories {
    mavenCentral()
}

intellij {
    version.set("2023.2.6") // The version of the IDE to build against
//    type.set("IC")          // "IC" = Community Edition, "IU" = Ultimate
    type.set("PC")
//    plugins.set(listOf("PythonCore"))
}

grammarKit {
    jflexRelease.set("1.9.1")
}

dependencies {
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(17)
}

tasks.test {
    useJUnitPlatform()
}

tasks {
    // Tell Gradle to generate the Lexer before compiling Kotlin
    generateLexer {
        // Source file (created in Step 2 of previous response)
        sourceFile.set(file("src/main/kotlin/it/kebula/plugin/Nemantix.flex"))

        // Output directory for the generated Java file
        targetOutputDir.set(file("src/main/gen/it/kebula/plugin"))

        // Clean up the lexer file when running 'clean'
        purgeOldFiles.set(true)
    }

    patchPluginXml {
        // "sinceBuild" defines the minimum IDE version (232 = 2023.2)
        sinceBuild.set("232")

        // "untilBuild" defines the maximum. Setting it to null means "Support all future versions"
        untilBuild.set("")

        changeNotes.set(provider {
            changelog.renderItem(
                changelog.getOrNull(project.version.toString()) ?: changelog.getUnreleased(),
                org.jetbrains.changelog.Changelog.OutputType.HTML
            )
        })
    }

    compileKotlin {
        dependsOn(generateLexer)
    }
}

// Add the generated source folder to the source set so IntelliJ sees it
sourceSets["main"].java.srcDir("src/main/gen")