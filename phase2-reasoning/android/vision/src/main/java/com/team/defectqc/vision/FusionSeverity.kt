
package com.team.defectqc.vision

import kotlin.math.max

/**
 * Fusion of detector + classifier for one image. Matches phase1-vision/fusion.py:
 *
 * 0. not_steel gate (classifier v2 only, checked first): top-1 == not_steel with
 *    p >= [VisionConfig.NOT_STEEL_THRESHOLD] -> "not a steel surface", alert false, no severity,
 *    decision RETAKE, detector boxes ignored. A not_steel top-1 below that threshold names no
 *    defect: the classifier does not fire and the detector decides alone.
 * 1-4. Phase 1.1 rule (kept detections, final class, alert, severity / Review), unchanged.
 *
 * Then the app-only **unsure -> HOLD** safety gate (not in fusion.py): on an alert, if
 *  (a) the classifier did not fire AND the best kept box is < [VisionConfig.UNSURE_BEST_BOX_CONFIDENCE], or
 *  (b) the kept boxes cover >= [VisionConfig.UNSURE_DISTINCT_CLASSES] different classes,
 * the result keeps its class / boxes / severity for display but gets decision HOLD
 * ("Unsure - manual check") and no Llama advice.
 */
object FusionSeverity {

    private const val CLASSIFIER_THRESHOLD = VisionConfig.CLASSIFIER_THRESHOLD

    private val classThresholds = VisionConfig.DETECTOR_CLASS_THRESHOLDS

    fun combine(
        detectorBoxes: List<VisionBox>,
        classifierProbabilities: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        detectorName: String = "fp32",
        detectorMs: Long = 0L,
        classifierMs: Long = 0L,
        classNames: List<String> = VisionConfig.NEU_CLASSES,
        notSteelThreshold: Float = VisionConfig.NOT_STEEL_THRESHOLD
    ): VisionResult {

        require(classNames == VisionConfig.NEU_CLASSES || classNames == VisionConfig.CLS7_CLASSES) {
            "unsupported classifier classes $classNames"
        }
        require(classifierProbabilities.size == classNames.size) {
            "classifier output has ${classifierProbabilities.size} values, expected ${classNames.size}"
        }
        require(imageWidth > 0 && imageHeight > 0)
        require(notSteelThreshold in 0f..1f)

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

        val classifier = ClassifierResult(
            classifierClass,
            classifierConfidence,
            classifierProbabilities
        )

        // 0. not_steel gate: reject the photo, ignore the detector.
        if (classifierClass == VisionConfig.NOT_STEEL && classifierConfidence >= notSteelThreshold) {
            return VisionResult(
                detector = detectorName,
                boxes = emptyList(),
                classifier = classifier,
                alert = false,
                finalClass = VisionConfig.NOT_STEEL_LABEL,
                decidedBy = "classifier",
                severityLevel = null,
                largestAreaPct = null,
                reason = "classifier ${VisionConfig.NOT_STEEL} %.3f >= %.2f: not a steel surface, %d kept detection(s) ignored".format(
                    java.util.Locale.US, classifierConfidence, notSteelThreshold, keptBoxes.size
                ),
                detectorMs = detectorMs,
                classifierMs = classifierMs,
                decision = VisionDecision.RETAKE
            )
        }

        // A not_steel top-1 below its threshold names no defect: the classifier does not fire.
        val classifierAccepted =
            classifierClass != VisionConfig.NOT_STEEL &&
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
                classifier = classifier,
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
            return unsureGate(classifierAccepted, keptBoxes, VisionResult(
                detector = detectorName,
                boxes = keptBoxes,
                classifier = classifier,
                alert = true,
                finalClass = finalClass,
                decidedBy = decidedBy,
                severityLevel = "Review",
                largestAreaPct = null,
                reason = "$decidedBy selected $finalClass; no kept " +
                    "$finalClass boxes are available for severity assessment.",
                detectorMs = detectorMs,
                classifierMs = classifierMs
            ))
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

        return unsureGate(classifierAccepted, keptBoxes, VisionResult(
            detector = detectorName,
            boxes = keptBoxes,
            classifier = classifier,
            alert = true,
            finalClass = finalClass,
            decidedBy = decidedBy,
            severityLevel = severity,
            largestAreaPct = largestAreaPct,
            reason = reason,
            detectorMs = detectorMs,
            classifierMs = classifierMs
        ))
    }

    /** App safety gate: an alert the models are unsure about -> decision HOLD (see class doc). */
    internal fun unsureGate(
        classifierFired: Boolean,
        keptBoxes: List<VisionBox>,
        result: VisionResult
    ): VisionResult {
        if (!result.alert || keptBoxes.isEmpty()) return result
        val best = keptBoxes.maxOf { it.confidence }
        val distinct = keptBoxes.map { it.className }.distinct()
        val why = when {
            distinct.size >= VisionConfig.UNSURE_DISTINCT_CLASSES ->
                "kept boxes of ${distinct.size} different classes $distinct"
            !classifierFired && best < VisionConfig.UNSURE_BEST_BOX_CONFIDENCE ->
                "classifier below its threshold and best kept box %.3f < %.2f".format(
                    java.util.Locale.US, best, VisionConfig.UNSURE_BEST_BOX_CONFIDENCE
                )
            else -> return result
        }
        return result.copy(
            decision = VisionDecision.HOLD,
            reason = "Unsure - manual check: $why; ${result.reason}"
        )
    }
}
