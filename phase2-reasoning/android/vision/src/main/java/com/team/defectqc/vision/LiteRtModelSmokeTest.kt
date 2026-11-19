
package com.team.defectqc.vision

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class VisionModelSmokeResult(
    val detectorInputShape: List<Int>,
    val detectorOutputShape: List<Int>,
    val classifierInputShape: List<Int>,
    val classifierOutputShape: List<Int>
)

class LiteRtModelSmokeTest(context: Context) {

    private val appContext = context.applicationContext

    fun run(): VisionModelSmokeResult {

        val detector = loadModel(
            "detector/defect_yolov8n_320_fp32.tflite"
        )

        try {
            val classifier = loadModel(
                "classifier/defect_cls_yolov8n_224_fp32.tflite"
            )

            try {
                runZeroInput(detector)
                runZeroInput(classifier)

                val detectorInput =
                    detector.getInputTensor(0).shape().toList()

                val detectorOutput =
                    detector.getOutputTensor(0).shape().toList()

                val classifierInput =
                    classifier.getInputTensor(0).shape().toList()

                val classifierOutput =
                    classifier.getOutputTensor(0).shape().toList()

                check(detectorInput == listOf(1, 3, 320, 320))
                check(detectorOutput == listOf(1, 10, 2100))

                check(classifierInput == listOf(1, 3, 224, 224))
                check(classifierOutput == listOf(1, 6))

                return VisionModelSmokeResult(
                    detectorInputShape = detectorInput,
                    detectorOutputShape = detectorOutput,
                    classifierInputShape = classifierInput,
                    classifierOutputShape = classifierOutput
                )

            } finally {
                classifier.close()
            }

        } finally {
            detector.close()
        }
    }

    private fun loadModel(assetPath: String): Interpreter {

        val modelFile = File(
            appContext.cacheDir,
            assetPath.substringAfterLast("/")
        )

        appContext.assets.open(assetPath).use { input ->
            modelFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }

        val options = Interpreter.Options()
            .setNumThreads(2)

        return Interpreter(modelFile, options)
    }

    private fun runZeroInput(interpreter: Interpreter) {

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)

        val input = ByteBuffer
            .allocateDirect(inputTensor.numBytes())
            .order(ByteOrder.nativeOrder())

        val output = ByteBuffer
            .allocateDirect(outputTensor.numBytes())
            .order(ByteOrder.nativeOrder())

        // Explicitly initialize dummy input with zeros.
        repeat(inputTensor.numBytes()) {
            input.put(0)
        }

        input.rewind()

        interpreter.run(input, output)
    }
}
