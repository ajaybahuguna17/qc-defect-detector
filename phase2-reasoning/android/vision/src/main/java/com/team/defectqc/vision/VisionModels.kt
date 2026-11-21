package com.team.defectqc.vision

data class VisionBox(
    val className: String,
    val classIndex: Int,
    val confidence: Float,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
)

data class ClassifierResult(
    val className: String,
    val confidence: Float,
    val probabilities: FloatArray
)

/** Gate decisions that override the severity -> action mapping (strings match fusion.py). */
object VisionDecision {
    /** Not a usable steel photo (not_steel classifier gate or a failed [ImageCheck]). */
    const val RETAKE = "RETAKE"

    /** Unsure result (app safety gate): manual check, no Llama. */
    const val HOLD = "HOLD"
}

data class VisionResult(
    val detector: String,
    val boxes: List<VisionBox>,
    /** Null when the models were not run (failed image check). */
    val classifier: ClassifierResult?,
    val alert: Boolean,
    val finalClass: String,
    val decidedBy: String,
    val severityLevel: String?,
    val largestAreaPct: Float?,
    val reason: String,
    val detectorMs: Long,
    val classifierMs: Long,
    /** [VisionDecision.RETAKE], [VisionDecision.HOLD] or null (decision from severity). */
    val decision: String? = null
)
