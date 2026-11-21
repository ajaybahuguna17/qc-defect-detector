package com.team.defectqc.advisor

import com.team.defectqc.core.Advisory
import com.team.defectqc.core.DetectionEvent
import org.junit.Assert.*
import org.junit.Test

class AdvisoryCacheTest {

    private fun event(
        defect: String = "scratches",
        severity: String? = "High",
        area: Float? = 10f,
        count: Int = 2,
        decidedBy: String = "classifier",
        reason: String = "Classifier selected scratches.",
        confidence: Float = 0.94f,
        decision: String? = null
    ) = DetectionEvent(
        id = "test-001",
        timestampEpochMs = 1L,
        imageName = "test.jpg",
        detector = "fp32",
        alert = true,
        finalClass = defect,
        decidedBy = decidedBy,
        classifierClass = defect,
        classifierConfidence = confidence,
        numBoxes = count,
        numFinalBoxes = count,
        largestAreaPct = area,
        severityLevel = severity,
        reason = reason,
        boxes = emptyList(),
        decision = decision
    )

    private fun advisory(id: String) = Advisory(
        eventId = id, finalClass = "scratches", severityLevel = "High", summary = "s", likelyCause = "c",
        recommendedAction = "a", llmMs = 1000L, tokensPerSecond = 10f, cached = false
    )

    private val cache = AdvisoryCache()

    @Test
    fun keyIsClassSeverityDecisionBucket() {
        assertEquals("scratches|high|none|2", cache.key(event()))
        assertEquals("crazing|review|hold|3+", cache.key(event("crazing", "Review", count = 5, decision = "HOLD")))
    }

    @Test
    fun normalizesDefectAndSeverity() {
        assertEquals(
            cache.key(event(defect = "Pitted Surface", severity = "High")),
            cache.key(event(defect = "pitted_surface", severity = "high"))
        )
    }

    @Test
    fun differentSeverityProducesDifferentKey() {
        assertNotEquals(cache.key(event(severity = "Low")), cache.key(event(severity = "High")))
    }

    @Test
    fun differentDecisionProducesDifferentKey() {
        assertNotEquals(cache.key(event(decision = null)), cache.key(event(decision = "HOLD")))
    }

    @Test
    fun differentClassProducesDifferentKey() {
        assertNotEquals(cache.key(event(defect = "scratches")), cache.key(event(defect = "patches")))
    }

    @Test
    fun countBuckets() {
        assertEquals("0", AdvisoryCache.bucket(0))
        assertEquals("1", AdvisoryCache.bucket(1))
        assertEquals("2", AdvisoryCache.bucket(2))
        assertEquals("3+", AdvisoryCache.bucket(3))
        assertEquals("3+", AdvisoryCache.bucket(12))
        assertNotEquals(cache.key(event(count = 1)), cache.key(event(count = 2)))
        assertNotEquals(cache.key(event(count = 2)), cache.key(event(count = 3)))
        assertEquals(cache.key(event(count = 3)), cache.key(event(count = 7)))
    }

    @Test
    fun areaConfidenceReasonAndDecidedByAreNotInKey() {
        assertEquals(
            cache.key(event(area = 5f, confidence = 0.70f, reason = "Classifier selected scratches.", decidedBy = "classifier")),
            cache.key(event(area = 15f, confidence = 0.99f, reason = "Detector selected scratches.", decidedBy = "detector"))
        )
    }

    @Test
    fun lruEvictsLeastRecentlyUsed() {
        val c = AdvisoryCache(maxEntries = 2)
        c.put("a", advisory("a"))
        c.put("b", advisory("b"))
        assertNotNull(c.get("a"))         // a is now most recently used
        c.put("c", advisory("c"))         // evicts b
        assertNull(c.get("b"))
        assertNotNull(c.get("a"))
        assertNotNull(c.get("c"))
        assertEquals(2, c.size())
    }

    @Test
    fun defaultBoundIsFifty() {
        val c = AdvisoryCache()
        repeat(60) { c.put("k$it", advisory("$it")) }
        assertEquals(50, c.size())
        assertNull(c.get("k0"))
        assertNotNull(c.get("k59"))
    }
}
