plugins {
    java
    id("com.github.ben-manes.versions") version "0.36.0"
    id("com.diffplug.spotless") version "5.9.0"
}

group = "org.jbduncan"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

object Versions {
    // Languages
    val jvmTarget = JavaVersion.VERSION_11

    // Dependencies
    const val guava = "30.1-jre"

    // Test dependencies
    const val junitJupiter = "5.7.0"
    const val truth = "1.1"
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(Versions.jvmTarget.majorVersion))
    }
}

dependencies {
    implementation("com.google.guava:guava:${Versions.guava}")

    testImplementation("org.junit.jupiter:junit-jupiter:${Versions.junitJupiter}") {
        because("this imports JUnit 5")
    }
    testImplementation("com.google.truth:truth:${Versions.truth}")
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
    }
    jvmArgs("-Xms1024m", "-Xmx1024m")
}

spotless {
    java {
        googleJavaFormat("1.9")
    }
}
