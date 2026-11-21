package com.team.defectqc.advisor

import com.team.defectqc.core.Advisory
import com.team.defectqc.core.DetectionEvent

/**
 * Bounded LRU cache of advice, keyed by what the advice is about:
 * final class + severity + gate decision + defects-found bucket (0, 1, 2, 3+).
 *
 * Raw reason text, box area and classifier confidence are deliberately NOT in the key: they change
 * on every photo and would make the cache useless. The decision and severity are rule-based and
 * shown by the app, never taken from the advice text.
 */
class AdvisoryCache(private val maxEntries: Int = DEFAULT_MAX_ENTRIES) {

    init {
        require(maxEntries > 0)
    }

    private val entries = object : LinkedHashMap<String, Advisory>(16, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Advisory>?): Boolean =
            size > maxEntries
    }

    fun key(event: DetectionEvent): String {
        val defect = DefectKnowledgeBase.normalize(event.finalClass)
        val severity = event.severityLevel?.trim()?.lowercase() ?: "none"
        val decision = event.decision?.trim()?.lowercase() ?: "none"
        return listOf(defect, severity, decision, bucket(event.numBoxes)).joinToString("|")
    }

    @Synchronized
    fun get(key: String): Advisory? = entries[key]

    @Synchronized
    fun put(key: String, advisory: Advisory) {
        entries[key] = advisory
    }

    @Synchronized
    fun clear() = entries.clear()

    @Synchronized
    fun size(): Int = entries.size

    companion object {
        const val DEFAULT_MAX_ENTRIES = 50

        /** Defects found (kept boxes, as shown in the app): "0", "1", "2" or "3+". */
        fun bucket(numBoxes: Int): String = when {
            numBoxes <= 0 -> "0"
            numBoxes >= 3 -> "3+"
            else -> numBoxes.toString()
        }
    }
}
