package com.team.defectqc.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FusionSeverityTest {

    private val W = 200
    private val H = 200
    private val NEU = VisionConfig.NEU_CLASSES
    private val CLS7 = VisionConfig.CLS7_CLASSES

    /** Full-width box with area ratio [ratio] of the 200x200 image. */
    private fun box(cls: String, conf: Float, ratio: Float = 0.01f, y0: Float = 0f) =
        VisionBox(cls, NEU.indexOf(cls), conf, 0f, y0, W.toFloat(), y0 + ratio * H)

    private fun probs(classes: List<String>, top: String, p: Float): FloatArray {
        val v = FloatArray(classes.size) { (1f - p) / (classes.size - 1) }
        v[classes.indexOf(top)] = p
        return v
    }

    private fun fuse7(boxes: List<VisionBox>, top: String, p: Float, nsThr: Float = 0.50f) =
        FusionSeverity.combine(boxes, probs(CLS7, top, p), W, H, classNames = CLS7, notSteelThreshold = nsThr)

    private fun fuse6(boxes: List<VisionBox>, top: String, p: Float) =
        FusionSeverity.combine(boxes, probs(NEU, top, p), W, H)

    // ---- not_steel gate ----

    @Test
    fun confidentNotSteelIsRetake() {
        val r = fuse7(emptyList(), VisionConfig.NOT_STEEL, 0.9f)
        assertEquals("not a steel surface", r.finalClass)
        assertFalse(r.alert)
        assertNull(r.severityLevel)
        assertEquals(VisionDecision.RETAKE, r.decision)
        assertEquals("classifier", r.decidedBy)
    }

    @Test
    fun notSteelIgnoresDetectorBoxes() {
        val r = fuse7(listOf(box("inclusion", 0.99f, 0.5f), box("scratches", 0.95f, 0.2f, 100f)), VisionConfig.NOT_STEEL, 0.8f)
        assertEquals(VisionDecision.RETAKE, r.decision)
        assertTrue(r.boxes.isEmpty())
        assertTrue(r.reason.contains("2 kept detection(s) ignored"))
    }

    @Test
    fun notSteelThresholdIsInclusive() {
        assertEquals(VisionDecision.RETAKE, fuse7(emptyList(), VisionConfig.NOT_STEEL, 0.70f, nsThr = 0.70f).decision)
        assertNull(fuse7(emptyList(), VisionConfig.NOT_STEEL, 0.6999f, nsThr = 0.70f).decision)
    }

    @Test
    fun tunedNotSteelThresholdIsUsedByDefault() {
        assertEquals(0.33f, VisionConfig.NOT_STEEL_THRESHOLD)
        val at = FusionSeverity.combine(emptyList(), probs(CLS7, VisionConfig.NOT_STEEL, 0.33f), W, H, classNames = CLS7)
        assertEquals(VisionDecision.RETAKE, at.decision)
        // 0.32 on not_steel: still top-1 (others share 0.68 / 6), but below t_ns -> no RETAKE
        val below = FusionSeverity.combine(emptyList(), probs(CLS7, VisionConfig.NOT_STEEL, 0.32f), W, H, classNames = CLS7)
        assertNull(below.decision)
    }

    @Test
    fun unsureNotSteelFallsBackToDetector() {
        // not_steel top-1 below t_ns but above the classifier threshold: names no defect
        val r = fuse7(listOf(box("scratches", 0.8f, 0.10f)), VisionConfig.NOT_STEEL, 0.85f, nsThr = 0.90f)
        assertEquals("scratches", r.finalClass)
        assertEquals("detector", r.decidedBy)
        assertTrue(r.alert)
        assertEquals("Medium", r.severityLevel)
        assertNull(r.decision)
    }

    @Test
    fun unsureNotSteelNoBoxesIsNoDefect() {
        val r = fuse7(emptyList(), VisionConfig.NOT_STEEL, 0.4f)
        assertEquals("no defect found", r.finalClass)
        assertFalse(r.alert)
        assertNull(r.decision)
    }

    @Test
    fun sevenClassDefectBehavesLikeSixClass() {
        val boxes = listOf(box("patches", 0.9f, 0.10f))
        val a = fuse7(boxes, "patches", 0.95f)
        val b = fuse6(boxes, "patches", 0.95f)
        assertEquals(b.finalClass, a.finalClass)
        assertEquals(b.severityLevel, a.severityLevel)
        assertEquals(b.decidedBy, a.decidedBy)
        assertNull(a.decision)
    }

    @Test(expected = IllegalArgumentException::class)
    fun wrongProbabilityCountThrows() {
        FusionSeverity.combine(emptyList(), FloatArray(7) { 1f / 7 }, W, H)   // 7 values, 6 classes
    }

    @Test(expected = IllegalArgumentException::class)
    fun unsupportedClassListThrows() {
        FusionSeverity.combine(emptyList(), FloatArray(7) { 1f / 7 }, W, H, classNames = NEU + "rust")
    }

    // ---- Phase 1.1 rule unchanged ----

    @Test
    fun classifierWithMatchingBoxGivesSeverity() {
        val r = fuse6(listOf(box("scratches", 0.9f, 0.10f)), "scratches", 0.95f)
        assertEquals("scratches", r.finalClass)
        assertEquals("Medium", r.severityLevel)
        assertNull(r.decision)
    }

    @Test
    fun classifierWithoutBoxIsReview() {
        val r = fuse6(emptyList(), "patches", 0.9f)
        assertEquals("Review", r.severityLevel)
        assertNull(r.decision)   // Review is not an unsure-HOLD: no kept boxes at all
    }

    @Test
    fun nothingIsNoDefect() {
        val r = fuse6(listOf(box("scratches", 0.29f)), "patches", 0.5f)
        assertEquals("no defect found", r.finalClass)
        assertFalse(r.alert)
    }

    @Test
    fun detectorThresholdsAreUnchanged() {
        assertEquals(
            mapOf("crazing" to 0.10f, "inclusion" to 0.50f, "patches" to 0.55f, "pitted_surface" to 0.15f,
                "rolled-in_scale" to 0.15f, "scratches" to 0.30f),
            VisionConfig.DETECTOR_CLASS_THRESHOLDS
        )
        assertEquals(0.64f, VisionConfig.CLASSIFIER_THRESHOLD)
    }

    // ---- unsure -> HOLD ----

    @Test
    fun weakDetectorOnlyIsHold() {
        val r = fuse6(listOf(box("crazing", 0.30f)), "inclusion", 0.5f)
        assertEquals("crazing", r.finalClass)
        assertTrue(r.alert)
        assertEquals(VisionDecision.HOLD, r.decision)
        assertTrue(r.reason.startsWith("Unsure - manual check"))
    }

    @Test
    fun bestBoxAtHalfIsNotUnsure() {
        val r = fuse6(listOf(box("crazing", 0.50f)), "inclusion", 0.5f)
        assertNull(r.decision)
        assertEquals("detector", r.decidedBy)
    }

    @Test
    fun weakBoxButClassifierFiredIsNotUnsure() {
        val r = fuse6(listOf(box("crazing", 0.20f)), "crazing", 0.9f)
        assertNull(r.decision)
        assertEquals("classifier", r.decidedBy)
    }

    @Test
    fun threeDistinctClassesIsHoldEvenWithConfidentClassifier() {
        val boxes = listOf(
            box("scratches", 0.9f, 0.05f), box("inclusion", 0.8f, 0.05f, 50f), box("patches", 0.7f, 0.05f, 100f)
        )
        val r = fuse6(boxes, "scratches", 0.95f)
        assertEquals(VisionDecision.HOLD, r.decision)
        assertTrue(r.reason.contains("3 different classes"))
        assertEquals(3, r.boxes.size)   // boxes kept for display
    }

    @Test
    fun twoDistinctConfidentClassesIsNotUnsure() {
        val r = fuse6(listOf(box("scratches", 0.9f, 0.05f), box("inclusion", 0.8f, 0.05f, 50f)), "scratches", 0.95f)
        assertNull(r.decision)
    }

    @Test
    fun weakOnlyDetectionWithSevenClassModelIsHold() {
        val r = fuse7(listOf(box("crazing", 0.2f)), VisionConfig.NOT_STEEL, 0.3f)
        assertEquals(VisionDecision.HOLD, r.decision)
    }
}
