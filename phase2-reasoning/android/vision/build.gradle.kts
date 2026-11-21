
plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

// Model assets (not in git, see phase1-vision/MODEL_IO.md):
//   detector/   always from phase1_1 (the detector is unchanged in Phase 1.2)
//   classifier/ from phase1_2 (7-class v2 with not_steel) if its files exist, else phase1_1 (6-class)
// Only .tflite, labels.txt and model_io.json are copied (no .pt checkpoints in the APK).
val exportedModels = file("../../../phase1-vision/models/exported")
val classifierSource: File = listOf("phase1_2", "phase1_1")
    .map { exportedModels.resolve("$it/classifier") }
    .firstOrNull { it.resolve("labels.txt").isFile && it.resolve("model_io.json").isFile }
    ?: exportedModels.resolve("phase1_1/classifier")
val visionAssetsDir = layout.buildDirectory.dir("generated/visionAssets")

val syncVisionAssets by tasks.registering(Sync::class) {
    into(visionAssetsDir)
    from(exportedModels.resolve("phase1_1/detector")) {
        include("*.tflite", "labels.txt", "model_io.json")
        into("detector")
    }
    from(classifierSource) {
        include("*.tflite", "labels.txt", "model_io.json")
        into("classifier")
    }
    doFirst { logger.lifecycle("vision assets: classifier from $classifierSource") }
}
tasks.named("preBuild") { dependsOn(syncVisionAssets) }

android {
    namespace = "com.team.defectqc.vision"
    compileSdk = 35

    defaultConfig {
        minSdk = 28
    }
    sourceSets {
        getByName("main") {
            // detector/ + classifier/ assembled by syncVisionAssets (above)
            assets.srcDir(visionAssetsDir)
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
