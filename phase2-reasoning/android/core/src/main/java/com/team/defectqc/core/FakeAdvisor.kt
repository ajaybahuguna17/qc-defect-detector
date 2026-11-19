package com.team.defectqc.core

/**
 * Deterministic fake advisor for UI and integration testing.
 */
class FakeAdvisor : Advisor {

    override suspend fun load(): Boolean = true

    override suspend fun advise(event: DetectionEvent): Advisory {

        require(event.alert) {
            "Clean inspections should bypass the advisor."
        }

        val matchingBoxes = event.boxes.filter {
            it.className.equals(event.finalClass, ignoreCase = true)
        }

        val isReview = event.severityLevel == "Review" ||
                matchingBoxes.isEmpty()

        val summary = if (isReview) {
            "${event.finalClass} flagged for operator review. ${event.reason}"
        } else {
            "${event.finalClass} detected with ${event.severityLevel} severity."
        }

        val action = if (isReview) {
            "Manually inspect the part and verify the reported defect before proceeding."
        } else {
            "Inspect the affected area and follow the plant's approved corrective procedure."
        }

        return Advisory(
            eventId = event.id,
            finalClass = event.finalClass,
            severityLevel = event.severityLevel,
            summary = summary,
            likelyCause = "Possible cause requires confirmation through process inspection.",
            recommendedAction = action,
            llmMs = 0L,
            tokensPerSecond = 0f,
            cached = false,
            rawResponse = "FAKE ADVISORY"
        )
    }

    override suspend fun ask(
        question: String,
        logSummary: String
    ): String = "Fake advisor: review the supplied inspection summary manually."

    override suspend fun close() = Unit
}