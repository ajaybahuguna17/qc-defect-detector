package com.example.executorchllamademo.aanetra.history

import com.team.defectqc.core.Advisory
import com.team.defectqc.core.DetectionEvent
import com.team.defectqc.core.QualityAction
import com.team.defectqc.core.QualityDecision

/**
 * One logged inspection (one row of the local `inspections` table).
 * Written after vision; the advice fields are filled in later when (if) advice arrives.
 */
data class InspectionRecord(
    val id: Long = 0L,
    val timestampMs: Long,
    /** [SOURCE_GALLERY] or [SOURCE_CAMERA]. */
    val source: String,
    /** Absolute path of the <= 256 px JPEG thumbnail in the app files dir, or null. */
    val thumbnailPath: String?,
    val defectType: String,
    val severity: String?,
    /** Operator decision label, e.g. "REJECT + STOP LINE", "RETAKE", "HOLD". */
    val decision: String,
    val decidedBy: String,
    val classifierConfidence: Float?,
    val defectsFound: Int,
    val visionMs: Long,
    val adviceText: String? = null,
    val adviceMs: Long? = null,
    val adviceCached: Boolean? = null,
    /** Why the photo was RETAKE or the part HOLD (rule text), else null. */
    val gateReason: String? = null
) {
    companion object {
        const val SOURCE_GALLERY = "gallery"
        const val SOURCE_CAMERA = "camera"

        /** Record for a finished vision result (no advice yet). Nothing is recomputed here. */
        fun fromInspection(
            event: DetectionEvent,
            decision: QualityDecision,
            visionMs: Long,
            source: String,
            thumbnailPath: String?
        ): InspectionRecord {
            val gated = decision.action == QualityAction.RETAKE || decision.action == QualityAction.HOLD
            return InspectionRecord(
                timestampMs = event.timestampEpochMs,
                source = source,
                thumbnailPath = thumbnailPath,
                defectType = event.finalClass,
                severity = event.severityLevel,
                decision = decision.label,
                decidedBy = event.decidedBy,
                classifierConfidence = event.classifierConfidence,
                defectsFound = event.numBoxes,
                visionMs = visionMs,
                gateReason = if (gated) event.reason else null
            )
        }

        /** The advice as stored: the three parsed sections. */
        fun adviceText(a: Advisory): String =
            "SUMMARY: ${a.summary}\nCAUSE: ${a.likelyCause}\nACTION: ${a.recommendedAction}"
    }
}
