package com.example.executorchllamademo

data class DefectContext(
    val defectName: String,
    val confidence: Float,
    val severity: String,
    val decision: String,

    val defectCount: Int,
    val bboxAreaRatio: Float,

    val surfaceType: String,
    val alert: Boolean,
    val reason: String,

    val lineId: String? = null,
    val lineSpeedMpm: Float? = null
)