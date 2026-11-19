package com.team.defectqc.advisor

import com.team.defectqc.core.Advisory
import com.team.defectqc.core.DetectionEvent

object AdvisoryParser {

    fun parse(
        event: DetectionEvent,
        raw: String,
        llmMs: Long,
        tokensPerSecond: Float,
        cached: Boolean = false
    ): Advisory {

        val fields = mutableMapOf<String, String>()

        raw.lineSequence().forEach { line ->
            val match = Regex(
                "^\\s*(SUMMARY|CAUSE|ACTION)\\s*:\\s*(.*)$",
                RegexOption.IGNORE_CASE
            ).find(line)

            if (match != null) {
                fields[match.groupValues[1].uppercase()] =
                    match.groupValues[2].trim()
            }
        }

        val matchingBoxes = event.boxes.filter {
            it.className.equals(event.finalClass, ignoreCase = true)
        }

        val isReview = event.severityLevel.equals(
            "Review",
            ignoreCase = true
        ) || event.numFinalBoxes == 0 || matchingBoxes.isEmpty()

        val reference = DefectKnowledgeBase.get(event.finalClass)

        val causeFallback = if (isReview) {
            "Cause is undetermined because no matching detector box is available. Manual verification is required."
        } else {
            "Possible cause (not confirmed): " +
                    (reference?.possibleCauses?.firstOrNull()
                        ?: "Further process investigation is required.")
        }

        val actionFallback = if (isReview) {
            "Manually inspect the part and verify the reported defect before proceeding."
        } else {
            reference?.correctiveActions?.firstOrNull()
                ?: "Inspect the affected area and follow the approved plant procedure."
        }

        val summaryFallback = if (isReview) {
            "${event.finalClass} flagged for operator review. ${event.reason}"
        } else {
            "${event.finalClass} detected with ${event.severityLevel} severity."
        }

        return Advisory(
            eventId = event.id,
            finalClass = event.finalClass,
            severityLevel = event.severityLevel,

            summary = fields["SUMMARY"].takeUnless { it.isNullOrBlank() }
                ?: summaryFallback,

            likelyCause = fields["CAUSE"].takeUnless { it.isNullOrBlank() }
                ?: causeFallback,

            recommendedAction = fields["ACTION"].takeUnless { it.isNullOrBlank() }
                ?: actionFallback,

            llmMs = llmMs,
            tokensPerSecond = tokensPerSecond,
            cached = cached,
            rawResponse = raw
        )
    }
}