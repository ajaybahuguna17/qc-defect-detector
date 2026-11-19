
package com.team.defectqc.vision

import kotlin.math.max

object FusionSeverity {

    private const val CLASSIFIER_THRESHOLD = 0.64f

    private val classThresholds = mapOf(
        "crazing" to 0.10f,
        "inclusion" to 0.50f,
        "patches" to 0.55f,
        "pitted_surface" to 0.15f,
        "rolled-in_scale" to 0.15f,
        "scratches" to 0.30f
    )

    private val classNames = listOf(
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    )

    fun combine(
        detectorBoxes: List<VisionBox>,
        classifierProbabilities: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        detectorName: String = "fp32",
        detectorMs: Long = 0L,
        classifierMs: Long = 0L
    ): VisionResult {

        require(classifierProbabilities.size == 6)
        require(imageWidth > 0 && imageHeight > 0)

        // Apply the class-specific detector thresholds.
        val keptBoxes = detectorBoxes
            .filter { box ->
                box.className in classThresholds &&
                    box.confidence >= (classThresholds[box.className] ?: 1f)
            }
            .sortedWith(
                compareByDescending<VisionBox> { it.confidence }
                    .thenBy { it.classIndex }
                    .thenBy { it.x1 }
                    .thenBy { it.y1 }
                    .thenBy { it.x2 }
                    .thenBy { it.y2 }
            )

        // Find classifier top-1 result.
        var classifierIndex = 0

        for (i in 1 until classifierProbabilities.size) {
            if (classifierProbabilities[i] >
                classifierProbabilities[classifierIndex]
            ) {
                classifierIndex = i
            }
        }

        val classifierConfidence =
            classifierProbabilities[classifierIndex]

        val classifierClass = classNames[classifierIndex]

        val classifierAccepted =
            classifierConfidence >= CLASSIFIER_THRESHOLD

        val alert =
            keptBoxes.isNotEmpty() || classifierAccepted

        val finalClass: String
        val decidedBy: String

        when {
            classifierAccepted -> {
                finalClass = classifierClass
                decidedBy = "classifier"
            }

            keptBoxes.isNotEmpty() -> {
                finalClass = keptBoxes.first().className
                decidedBy = "detector"
            }

            else -> {
                finalClass = "no defect found"
                decidedBy = "none"
            }
        }

        val finalBoxes = keptBoxes.filter {
            it.className == finalClass
        }

        if (!alert) {
            return VisionResult(
                detector = detectorName,
                boxes = emptyList(),
                classifier = ClassifierResult(
                    classifierClass,
                    classifierConfidence,
                    classifierProbabilities
                ),
                alert = false,
                finalClass = finalClass,
                decidedBy = decidedBy,
                severityLevel = null,
                largestAreaPct = null,
                reason = "No defects detected.",
                detectorMs = detectorMs,
                classifierMs = classifierMs
            )
        }

        // An alert without any box for the final class requires review.
        if (finalBoxes.isEmpty()) {
            return VisionResult(
                detector = detectorName,
                boxes = keptBoxes,
                classifier = ClassifierResult(
                    classifierClass,
                    classifierConfidence,
                    classifierProbabilities
                ),
                alert = true,
                finalClass = finalClass,
                decidedBy = decidedBy,
                severityLevel = "Review",
                largestAreaPct = null,
                reason = "$decidedBy selected $finalClass; no kept " +
                    "$finalClass boxes are available for severity assessment.",
                detectorMs = detectorMs,
                classifierMs = classifierMs
            )
        }

        // Calculate the largest final-class box area as a fraction
        // of the original image area.
        val imageArea = imageWidth.toFloat() * imageHeight.toFloat()

        val largestAreaRatio = finalBoxes.maxOf { box ->
            val width = max(0f, box.x2 - box.x1)
            val height = max(0f, box.y2 - box.y1)
            (width * height) / imageArea
        }

        val largestAreaPct = largestAreaRatio * 100f

        var severity = when {
            largestAreaRatio < 0.082f -> "Low"
            largestAreaRatio < 0.184f -> "Medium"
            largestAreaRatio <= 0.359f -> "High"
            else -> "Critical"
        }

        // Three or more final-class detections raise severity by one level.
        if (finalBoxes.size >= 3) {
            severity = when (severity) {
                "Low" -> "Medium"
                "Medium" -> "High"
                "High" -> "Critical"
                else -> "Critical"
            }
        }

        val decisionReason = when (decidedBy) {
            "classifier" ->
                "classifier $finalClass %.3f >= %.2f".format(
                    java.util.Locale.US,
                    classifierConfidence,
                    CLASSIFIER_THRESHOLD
                )

            "detector" ->
                "highest-confidence kept detector selected $finalClass"

            else -> "no accepted prediction"
        }

        val reason = "$decisionReason; severity on " +
            "${finalBoxes.size} kept $finalClass box(es): " +
            "largest box area ratio %.4f -> %s".format(
                java.util.Locale.US,
                largestAreaRatio,
                severity
            ) +
            if (finalBoxes.size >= 3) "; escalated for 3+ detections" else ""

        return VisionResult(
            detector = detectorName,
            boxes = keptBoxes,
            classifier = ClassifierResult(
                classifierClass,
                classifierConfidence,
                classifierProbabilities
            ),
            alert = true,
            finalClass = finalClass,
            decidedBy = decidedBy,
            severityLevel = severity,
            largestAreaPct = largestAreaPct,
            reason = reason,
            detectorMs = detectorMs,
            classifierMs = classifierMs
        )
    }
}
