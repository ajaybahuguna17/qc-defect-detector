package com.team.defectqc.advisor

import com.team.defectqc.core.Advisory
import com.team.defectqc.core.DetectionEvent
import java.util.Locale

class AdvisoryCache {

    private val entries = LinkedHashMap<String, Advisory>()

    fun key(event: DetectionEvent): String {
        val defect = DefectKnowledgeBase.normalize(event.finalClass)

        val severity = event.severityLevel
            ?.trim()
            ?.lowercase() ?: "unknown"

        val area = event.largestAreaPct?.let {
            String.format(Locale.US, "%.2f", it)
        } ?: "unknown"

        val classifierConfidence = event.classifierConfidence?.let {
            String.format(Locale.US, "%.2f", it)
        } ?: "unknown"

        return listOf(
            defect,
            severity,
            event.decidedBy.lowercase(),
            event.numBoxes.toString(),
            event.numFinalBoxes.toString(),
            area,
            DefectKnowledgeBase.normalize(event.classifierClass.orEmpty()),
            classifierConfidence,
            event.reason.trim().lowercase()
        ).joinToString("|")
    }

    fun get(key: String): Advisory? = entries[key]

    fun put(key: String, advisory: Advisory) {
        entries[key] = advisory
    }

    fun clear() = entries.clear()

    fun size(): Int = entries.size
}