package com.team.defectqc.advisor

import com.team.defectqc.core.DetectionEvent
import org.junit.Assert.*
import org.junit.Test

class AdvisoryCacheTest {

    private fun event(
        defect: String = "scratches",
        severity: String = "High",
        area: Float? = 10f,
        count: Int = 2,
        decidedBy: String = "classifier",
        reason: String = "Classifier selected scratches."
    ) = DetectionEvent(
        id = "test-001",
        timestampEpochMs = 1L,
        imageName = "test.jpg",
        detector = "fp32",
        alert = true,
        finalClass = defect,
        decidedBy = decidedBy,
        classifierClass = defect,
        classifierConfidence = 0.94f,
        numBoxes = count,
        numFinalBoxes = count,
        largestAreaPct = area,
        severityLevel = severity,
        reason = reason,
        boxes = emptyList()
    )

    @Test
    fun normalizesDefectAndSeverity() {
        val cache = AdvisoryCache()

        val first = cache.key(
            event(defect = "Pitted Surface", severity = "High")
        )

        val second = cache.key(
            event(defect = "pitted_surface", severity = "high")
        )

        assertEquals(first, second)
    }

    @Test
    fun differentSeverityProducesDifferentKey() {
        val cache = AdvisoryCache()

        assertNotEquals(
            cache.key(event(severity = "Low")),
            cache.key(event(severity = "High"))
        )
    }

    @Test
    fun differentAffectedAreaProducesDifferentKey() {
        val cache = AdvisoryCache()

        assertNotEquals(
            cache.key(event(area = 5f)),
            cache.key(event(area = 15f))
        )
    }

    @Test
    fun differentDetectionCountsProduceDifferentKey() {
        val cache = AdvisoryCache()

        assertNotEquals(
            cache.key(event(count = 1)),
            cache.key(event(count = 3))
        )
    }

    @Test
    fun differentInspectionReasonsProduceDifferentKey() {
        val cache = AdvisoryCache()

        assertNotEquals(
            cache.key(event(reason = "Classifier selected scratches.")),
            cache.key(event(reason = "Detector selected scratches."))
        )
    }
}