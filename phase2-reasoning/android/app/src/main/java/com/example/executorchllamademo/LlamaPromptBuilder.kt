package com.example.executorchllamademo

import java.util.Locale

object LlamaPromptBuilder {

    fun buildPrompt(context: DefectContext): String {

        val defect = DefectKnowledgeBase.getDefect(context.defectName)

        val defectInformation = if (defect == null) {
            "Defect information is unavailable. Manual inspection is required."
        } else {
            """
            Description: ${defect.description}

            Possible causes:
            ${defect.possibleCauses.joinToString("\n") { "- $it" }}

            Corrective actions:
            ${defect.correctiveActions.joinToString("\n") { "- $it" }}

            Preventive measures:
            ${defect.preventiveMeasures.joinToString("\n") { "- $it" }}
            """.trimIndent()
        }

        return """
            You are a quality-control assistant in a steel factory.
            Be brief and practical. Use only the facts provided.

            INSPECTION DETAILS:
            Defect: ${context.defectName}
            Confidence: ${String.format(Locale.US, "%.2f", context.confidence)}
            Severity: ${context.severity}
            Quality decision: ${context.decision}
            Number of defects: ${context.defectCount}
            Affected area: ${String.format(Locale.US, "%.2f", context.bboxAreaRatio * 100)}%
            Surface type: ${context.surfaceType}
            Alert: ${context.alert}
            Inspection reason: ${context.reason}

            DEFECT REFERENCE:
            $defectInformation

            IMPORTANT RULES:
            1. Severity and quality decision are already determined by the rule-based system.
            2. Never change the supplied severity or quality decision.
            3. Possible causes are not confirmed causes.
            4. Never invent measurements or inspection results.
            5. If information is unavailable, say so.

            Respond in exactly these 3 lines:

            SUMMARY: Explain the detected defect and inspection result.
            CAUSE: Mention possible causes, not confirmed causes.
            ACTION: Suggest practical corrective actions.

        """.trimIndent()
    }
}