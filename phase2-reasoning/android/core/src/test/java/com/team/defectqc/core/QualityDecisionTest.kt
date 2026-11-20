package com.team.defectqc.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QualityDecisionTest {

    @Test
    fun noDefectPasses() {
        val d = QualityDecision.decide(alert = false, severityLevel = null)
        assertEquals(QualityAction.PASS, d.action)
        assertFalse(d.stopLine)
        assertEquals("PASS", d.label)
    }

    @Test
    fun noDefectPassesEvenIfSeverityIsSet() {
        assertEquals(QualityAction.PASS, QualityDecision.decide(false, "Critical").action)
    }

    @Test
    fun lowAccepts() {
        val d = QualityDecision.decide(true, "Low")
        assertEquals(QualityAction.ACCEPT, d.action)
        assertFalse(d.stopLine)
    }

    @Test
    fun mediumReworks() {
        assertEquals(QualityAction.REWORK, QualityDecision.decide(true, "Medium").action)
    }

    @Test
    fun highRejectsWithoutStoppingLine() {
        val d = QualityDecision.decide(true, "High")
        assertEquals(QualityAction.REJECT, d.action)
        assertFalse(d.stopLine)
        assertEquals("REJECT", d.label)
    }

    @Test
    fun criticalRejectsAndStopsLine() {
        val d = QualityDecision.decide(true, "Critical")
        assertEquals(QualityAction.REJECT, d.action)
        assertTrue(d.stopLine)
        assertEquals("REJECT + STOP LINE", d.label)
    }

    @Test
    fun reviewHolds() {
        val d = QualityDecision.decide(true, "Review")
        assertEquals(QualityAction.HOLD, d.action)
        assertFalse(d.stopLine)
    }

    @Test
    fun missingOrUnknownSeverityOnAlertHolds() {
        assertEquals(QualityAction.HOLD, QualityDecision.decide(true, null).action)
        assertEquals(QualityAction.HOLD, QualityDecision.decide(true, "Severe?").action)
    }

    @Test
    fun severityMatchIsCaseInsensitive() {
        assertEquals(QualityAction.REWORK, QualityDecision.decide(true, " medium ").action)
        assertTrue(QualityDecision.decide(true, "CRITICAL").stopLine)
    }

    @Test
    fun decideFromEventUsesAlertAndSeverity() {
        val event = DetectionEvent(
            id = "e1",
            timestampEpochMs = 0L,
            imageName = "img.jpg",
            detector = "fp32",
            alert = true,
            finalClass = "scratches",
            decidedBy = "detector",
            classifierClass = "scratches",
            classifierConfidence = 0.5f,
            numBoxes = 1,
            numFinalBoxes = 1,
            largestAreaPct = 20f,
            severityLevel = "High",
            reason = "test",
            boxes = emptyList()
        )
        assertEquals(QualityAction.REJECT, QualityDecision.decide(event).action)
    }
}
