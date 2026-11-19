
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.team.defectqc.vision"
    compileSdk = 35

    defaultConfig {
        minSdk = 28
    }
    sourceSets {
    getByName("main") {
        assets.srcDir(
            file("../../../phase1-vision/models/exported/phase1_1")
        )
    }
}

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    api(project(":core"))

    // On-device TensorFlow Lite model inference
    implementation("com.google.ai.edge.litert:litert:1.4.2")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    testImplementation("junit:junit:4.13.2")
}
