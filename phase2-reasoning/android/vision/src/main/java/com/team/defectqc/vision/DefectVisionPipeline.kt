
package com.team.defectqc.vision

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock

class DefectVisionPipeline(context: Context) : AutoCloseable {

    private val engine = LiteRtVisionEngine(context)

    @Synchronized
    fun analyze(bitmap: Bitmap): VisionResult {

        require(!bitmap.isRecycled) {
            "Cannot process a recycled bitmap"
        }

        // Step 1: Prepare both model inputs.
        val detectorInput =
            ImagePreprocessor.preprocessDetector(bitmap)

        val classifierInput =
            ImagePreprocessor.preprocessClassifier(bitmap)

        // Step 2: Run detector.
        val detectorStart = SystemClock.elapsedRealtime()

        val detectorOutput =
            engine.runDetector(detectorInput.tensor)

        val detectorMs =
            SystemClock.elapsedRealtime() - detectorStart

        // Step 3: Decode boxes and apply NMS.
        val boxes = DetectorDecoder.decode(
            detectorOutput,
            detectorInput
        )

        // Step 4: Run classifier.
        val classifierStart = SystemClock.elapsedRealtime()

        val classifierOutput =
            engine.runClassifier(classifierInput)

        val classifierMs =
            SystemClock.elapsedRealtime() - classifierStart

        // Step 5: Apply rule-based fusion and severity.
        return FusionSeverity.combine(
            detectorBoxes = boxes,
            classifierProbabilities = classifierOutput,
            imageWidth = bitmap.width,
            imageHeight = bitmap.height,
            detectorName = "fp32",
            detectorMs = detectorMs,
            classifierMs = classifierMs
        )
    }

    override fun close() {
        engine.close()
    }
}
