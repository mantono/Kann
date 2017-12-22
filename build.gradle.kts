import org.gradle.kotlin.dsl.extra
import org.jetbrains.kotlin.gradle.dsl.Coroutines
import org.jetbrains.kotlin.gradle.plugin.KotlinPluginWrapper
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

val kotlinVersion = plugins.getPlugin(KotlinPluginWrapper::class.java).kotlinPluginVersion
val kotlinCoroutinesVersion = "0.19.3"
val junitPlatformVersion = "1.0.1"

plugins {
    kotlin("jvm").version("1.2.10")
    application
    idea
}

application {
    group = "com.mantono.kann"
    version = "1.0-SNAPSHOT"
    description = "Kotlin Artificial Neural Network"
}

repositories {
    mavenCentral()
    jcenter()
}

dependencies {
    compile(kotlin("stdlib", kotlinVersion))
    compile(kotlin("reflect", kotlinVersion))
    compile("org.jetbrains.kotlinx:kotlinx-coroutines-core:$kotlinCoroutinesVersion")

    testCompile("org.junit.jupiter:junit-jupiter-api:5.0.1")
    testRuntime("org.junit.platform:junit-platform-launcher:$junitPlatformVersion")
    testRuntime("org.junit.jupiter:junit-jupiter-engine:5.0.1")
}

kotlin {
    experimental.coroutines = Coroutines.ENABLE
}

tasks {
    withType<KotlinCompile> {
        kotlinOptions.jvmTarget = "1.8"
    }
}
