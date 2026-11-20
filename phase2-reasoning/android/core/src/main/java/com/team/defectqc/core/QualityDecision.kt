package com.team.defectqc.core

/**
 * Line action for one inspected part.
 *
 * This is decided by a fixed rule from the vision result only.
 * Llama never changes it.
 */
enum class QualityAction {
    PASS,
    ACCEPT,
    REWORK,
    REJECT,
    HOLD
}

data class QualityDecision(
    val action: QualityAction,
    val stopLine: Boolean,
    val explanation: String
) {
    /** Short text for the operator, e.g. "REJECT + STOP LINE". */
    val label: String
        get() = if (stopLine) "${action.name} + STOP LINE" else action.name

    companion object {

        /**
         * Deterministic mapping:
         * no defect -> PASS, Low -> ACCEPT, Medium -> REWORK,
         * High -> REJECT, Critical -> REJECT + STOP LINE,
         * Review (or any unknown severity) -> HOLD for manual check.
         */
        fun decide(alert: Boolean, severityLevel: String?): QualityDecision {
            if (!alert) {
                return QualityDecision(QualityAction.PASS, false, "No defect found.")
            }

            return when (severityLevel?.trim()?.lowercase()) {
                "low" -> QualityDecision(
                    QualityAction.ACCEPT, false, "Low severity defect; part can be accepted."
                )
                "medium" -> QualityDecision(
                    QualityAction.REWORK, false, "Medium severity defect; send part for rework."
                )
                "high" -> QualityDecision(
                    QualityAction.REJECT, false, "High severity defect; reject the part."
                )
                "critical" -> QualityDecision(
                    QualityAction.REJECT, true, "Critical defect; reject the part and stop the line."
                )
                else -> QualityDecision(
                    QualityAction.HOLD, false, "Hold the part for a manual check."
                )
            }
        }

        fun decide(event: DetectionEvent): QualityDecision =
            decide(event.alert, event.severityLevel)
    }
}
