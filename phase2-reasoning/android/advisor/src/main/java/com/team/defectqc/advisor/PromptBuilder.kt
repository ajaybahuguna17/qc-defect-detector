package com.team.defectqc.advisor

import com.team.defectqc.core.DetectionEvent
import java.util.Locale

object PromptBuilder {

    const val SYSTEM_PROMPT =
        "You are a steel-factory quality-control assistant. " +
        "Explain inspection findings concisely and practically. " +
        "Use only the supplied inspection facts and reference notes. " +
        "Never change the detected defect class, alert status, or severity. " +
        "Never invent measurements or claim that a possible cause is confirmed. " +
        "You do not make ACCEPT, REWORK, or REJECT decisions. " +
        "For Review findings, recommend manual operator inspection. " +
        "Return exactly three lines and no introduction: " +
        "SUMMARY: ...\nCAUSE: ...\nACTION: ..."

    private fun wrapAsLlama3Chat(userText: String): String =
        "<|start_header_id|>system<|end_header_id|>\n\n" +
        SYSTEM_PROMPT + "<|eot_id|>" +
        "<|start_header_id|>user<|end_header_id|>\n\n" +
        userText + "<|eot_id|>" +
        "<|start_header_id|>assistant<|end_header_id|>\n\n"

    fun inspection(event: DetectionEvent): String {

        require(event.alert) {
            "Clean inspections must bypass the advisor."
        }

        val matchingBoxes = event.boxes.filter {
            it.className.equals(event.finalClass, ignoreCase = true)
        }

        val isReview = event.severityLevel == "Review" ||
                matchingBoxes.isEmpty()

        val reference = DefectKnowledgeBase.get(event.finalClass)

        val referenceNotes = if (reference == null) {
            "No reference entry is available. Recommend manual process investigation."
        } else {
            """
            Description: ${reference.description}
            Possible causes: ${reference.possibleCauses.joinToString("; ")}
            Corrective actions: ${reference.correctiveActions.joinToString("; ")}
            Preventive measures: ${reference.preventiveMeasures.joinToString("; ")}
            """.trimIndent()
        }

        val boxDetails = if (matchingBoxes.isEmpty()) {
            "No kept bounding box belonging to the final defect class."
        } else {
            matchingBoxes
                .sortedByDescending { it.confidence }
                .take(3)
                .joinToString("\n") { box ->
                    "Class: ${box.className}, " +
                    "confidence: ${String.format(Locale.US, "%.3f", box.confidence)}, " +
                    "coordinates: ${box.bboxXYXY}"
                }
        }

        val body = """
            INSPECTION RESULT:

            Event ID: ${event.id}
            Image: ${event.imageName}
            Detector: ${event.detector}

            Alert: ${event.alert}
            Final defect class: ${event.finalClass}
            Class selected by: ${event.decidedBy}
            Severity from deterministic rules: ${event.severityLevel ?: "Not available"}

            Classifier class: ${event.classifierClass ?: "Not available"}
            Classifier confidence: ${event.classifierConfidence ?: "Not available"}

            Total kept detections: ${event.numBoxes}
            Kept detections of final class: ${event.numFinalBoxes}
            Largest affected area: ${event.largestAreaPct?.let {
                String.format(Locale.US, "%.2f%%", it)
            } ?: "Not available"}

            Inspection reason: ${event.reason}

            FINAL CLASS DETECTIONS:
            $boxDetails

            Review required: $isReview

            REFERENCE NOTES:
            $referenceNotes

            Explain the supplied inspection result.
            Do not change the final class or severity.
            Do not make an acceptance or rejection decision.
            Treat possible causes as unconfirmed.
            For Review findings, emphasize manual operator verification.
            Return exactly the required three-line format.
        """.trimIndent()

        return wrapAsLlama3Chat(body)
    }

    fun question(question: String, logSummary: String): String {

        val body = """
            OPERATOR QUESTION:
            $question

            RECENT INSPECTION LOG SUMMARY:
            $logSummary

            Answer using only the supplied summary.
            If information is insufficient, state what is missing.
            Do not invent counts, trends, causes, or measurements.
            Keep the answer concise.
        """.trimIndent()

        return wrapAsLlama3Chat(body)
    }
}