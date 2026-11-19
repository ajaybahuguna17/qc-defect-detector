
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

data class VisionResult(
    val detector: String,
    val boxes: List<VisionBox>,
    val classifier: ClassifierResult,
    val alert: Boolean,
    val finalClass: String,
    val decidedBy: String,
    val severityLevel: String?,
    val largestAreaPct: Float?,
    val reason: String,
    val detectorMs: Long,
    val classifierMs: Long
)
