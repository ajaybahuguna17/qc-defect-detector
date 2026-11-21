
package com.team.defectqc.vision

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock

class DefectVisionPipeline(context: Context) : AutoCloseable {

    private val engine = LiteRtVisionEngine(context)

    /** The classifier in the assets (6-class v1 or 7-class v2 with not_steel). */
    val classifierSpec: ClassifierSpec get() = engine.classifierSpec

    @Synchronized
    fun analyze(bitmap: Bitmap): VisionResult {

        require(!bitmap.isRecycled) {
            "Cannot process a recycled bitmap"
        }

        // Step 0: input checks BEFORE vision. A failed check -> RETAKE, models not run.
        val check = checkImage(bitmap)
        if (!check.passed) {
            return VisionResult(
                detector = "fp32",
                boxes = emptyList(),
                classifier = null,
                alert = false,
                finalClass = VisionConfig.IMAGE_CHECK_LABEL,
                decidedBy = "image_check",
                severityLevel = null,
                largestAreaPct = null,
                reason = check.reason,
                detectorMs = 0L,
                classifierMs = 0L,
                decision = VisionDecision.RETAKE
            )
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
            classifierMs = classifierMs,
            classNames = engine.classifierSpec.classes
        )
    }

    private fun checkImage(bitmap: Bitmap): ImageCheck.ImageCheckResult {
        val cfg = VisionConfig.IMAGE_CHECK
        if (minOf(bitmap.width, bitmap.height) < cfg.minSidePx) {
            return ImageCheck.check(bitmap.width, bitmap.height, IntArray(0), 0, 0, cfg)
        }
        // Large photos: shrink with the platform filter first to bound memory; ImageCheck then
        // makes its own 256 px copy (box average), as in the JVM test.
        val longest = maxOf(bitmap.width, bitmap.height)
        val src = if (longest > PRESCALE_SIDE) {
            val s = PRESCALE_SIDE.toFloat() / longest
            Bitmap.createScaledBitmap(
                bitmap,
                maxOf(1, Math.round(bitmap.width * s)),
                maxOf(1, Math.round(bitmap.height * s)),
                true
            )
        } else bitmap
        try {
            val px = IntArray(src.width * src.height)
            src.getPixels(px, 0, src.width, 0, 0, src.width, src.height)
            return ImageCheck.check(bitmap.width, bitmap.height, px, src.width, src.height, cfg)
        } finally {
            if (src !== bitmap) src.recycle()
        }
    }

    override fun close() {
        engine.close()
    }

    private companion object {
        const val PRESCALE_SIDE = 1024
    }
}
