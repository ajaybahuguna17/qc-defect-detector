
package com.team.defectqc.vision

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class LiteRtVisionEngine(context: Context) {

    private val appContext = context.applicationContext

    private val detector: Interpreter
    private val classifier: Interpreter

    init {
        detector = loadModel(
            "detector/defect_yolov8n_320_fp32.tflite"
        )

        classifier = loadModel(
            "classifier/defect_cls_yolov8n_224_fp32.tflite"
        )

        checkModelShapes()
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

    private fun checkModelShapes() {

        check(
            detector.getInputTensor(0).shape().toList() ==
                listOf(1, 3, 320, 320)
        )

        check(
            detector.getOutputTensor(0).shape().toList() ==
                listOf(1, 10, 2100)
        )

        check(
            classifier.getInputTensor(0).shape().toList() ==
                listOf(1, 3, 224, 224)
        )

        check(
            classifier.getOutputTensor(0).shape().toList() ==
                listOf(1, 6)
        )

        check(detector.getInputTensor(0).dataType() == DataType.FLOAT32)
        check(classifier.getInputTensor(0).dataType() == DataType.FLOAT32)
    }

    fun runDetector(inputData: FloatArray): FloatArray {

        require(inputData.size == 3 * 320 * 320)

        val input = createInputBuffer(inputData)
        val output = ByteBuffer
            .allocateDirect(10 * 2100 * 4)
            .order(ByteOrder.nativeOrder())

        synchronized(detector) {
            detector.run(input, output)
        }

        output.rewind()

        return FloatArray(10 * 2100) {
            output.float
        }
    }

    fun runClassifier(inputData: FloatArray): FloatArray {

        require(inputData.size == 3 * 224 * 224)

        val input = createInputBuffer(inputData)
        val output = ByteBuffer
            .allocateDirect(6 * 4)
            .order(ByteOrder.nativeOrder())

        synchronized(classifier) {
            classifier.run(input, output)
        }

        output.rewind()

        return FloatArray(6) {
            output.float
        }
    }

    private fun createInputBuffer(data: FloatArray): ByteBuffer {

        val buffer = ByteBuffer
            .allocateDirect(data.size * 4)
            .order(ByteOrder.nativeOrder())

        data.forEach { buffer.putFloat(it) }

        buffer.rewind()
        return buffer
    }

    fun close() {
        detector.close()
        classifier.close()
    }
}
