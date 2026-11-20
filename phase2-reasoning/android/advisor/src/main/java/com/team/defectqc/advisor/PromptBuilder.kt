
package com.team.defectqc.advisor

import com.team.defectqc.core.DetectionEvent
import java.util.Locale

object PromptBuilder {

    private const val SYSTEM_PROMPT =
        "You explain steel inspection results. " +
        "Inspection facts and rule-based class, alert and severity are final. " +
        "Never change them or invent measurements. " +
        "Causes are possibilities, not confirmed facts. " +
        "Do not decide ACCEPT, REWORK or REJECT. " +
        "If review is required, recommend manual inspection. " +
        "Use exactly three lines:\n" +
        "SUMMARY: ...\nCAUSE: ...\nACTION: ..."

    private const val QUESTION_SYSTEM_PROMPT =
        "You are a steel-factory quality-control assistant. " +
        "Answer using only the supplied inspection summary. " +
        "Do not invent counts, trends, causes or measurements. " +
        "If information is missing, say so. Be concise."

    private fun wrapAsLlama3Chat(
        systemText: String,
        userText: String
    ): String =
        "<|start_header_id|>system<|end_header_id|>\n\n" +
        systemText + "<|eot_id|>" +
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
            "No reference available. Recommend manual investigation."
        } else {
            "Description: ${reference.description}\n" +
            "Possible causes: ${reference.possibleCauses.joinToString("; ")}\n" +
            "Corrective actions: ${reference.correctiveActions.joinToString("; ")}"
        }

        val confidence = event.classifierConfidence?.let {
            String.format(Locale.US, "%.3f", it)
        } ?: "N/A"

        val area = event.largestAreaPct?.let {
            String.format(Locale.US, "%.2f%%", it)
        } ?: "N/A"

        val body =
            "INSPECTION:\n" +
            "Alert: ${event.alert}\n" +
            "Defect: ${event.finalClass}\n" +
            "Severity: ${event.severityLevel ?: "N/A"}\n" +
            "Classifier confidence: $confidence\n" +
            "Detections: ${event.numBoxes} total, " +
            "${event.numFinalBoxes} of final class\n" +
            "Largest affected area: $area\n" +
            "Manual review required: $isReview\n\n" +
            "REFERENCE:\n" +
            referenceNotes

        return wrapAsLlama3Chat(SYSTEM_PROMPT, body)
    }

    fun question(question: String, logSummary: String): String {

        val body =
            "OPERATOR QUESTION:\n$question\n\n" +
            "INSPECTION LOG SUMMARY:\n$logSummary"

        return wrapAsLlama3Chat(QUESTION_SYSTEM_PROMPT, body)
    }
}
