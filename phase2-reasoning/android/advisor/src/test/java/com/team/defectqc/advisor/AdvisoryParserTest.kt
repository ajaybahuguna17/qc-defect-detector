package com.team.defectqc.advisor

import com.team.defectqc.core.*
import org.junit.Assert.*
import org.junit.Test

class AdvisoryParserTest {

    private fun event() = DetectionEvent(
        id = "e1",
        timestampEpochMs = 1L,
        imageName = "test.jpg",
        detector = "fp32",
        alert = true,
        finalClass = "scratches",
        decidedBy = "classifier",
        classifierClass = "scratches",
        classifierConfidence = 0.94f,
        numBoxes = 1,
        numFinalBoxes = 1,
        largestAreaPct = 5f,
        severityLevel = "Medium",
        reason = "Classifier selected scratches.",
        boxes = listOf(
            DefectBox(
                "scratches",
                0.94f,
                listOf(10f, 20f, 30f, 40f)
            )
        )
    )

    @Test
    fun parsesThreeLines() {
        val e = event()

        val a = AdvisoryParser.parse(
            e,
            "SUMMARY: Scratches found.\n" +
            "CAUSE: Possible roller contact.\n" +
            "ACTION: Inspect rollers.",
            100,
            5f
        )

        assertEquals("Scratches found.", a.summary)
        assertEquals("Possible roller contact.", a.likelyCause)
        assertEquals("Inspect rollers.", a.recommendedAction)
        assertEquals("scratches", a.finalClass)
        assertEquals("Medium", a.severityLevel)
    }

    @Test
    fun fillsMissingFieldsFromKnowledgeBase() {
        val e = event()

        val a = AdvisoryParser.parse(
            e,
            "SUMMARY: Scratches found.",
            100,
            0f
        )

        assertTrue(a.likelyCause.isNotBlank())
        assertTrue(a.recommendedAction.isNotBlank())
    }

    @Test
    fun ignoresIntroductoryText() {
        val e = event()

        val a = AdvisoryParser.parse(
            e,
            "Here is the requested information:\n" +
            "SUMMARY: Scratches found.\n" +
            "CAUSE: Abrasive contact.\n" +
            "ACTION: Inspect guides.",
            100,
            0f
        )

        assertEquals("Scratches found.", a.summary)
    }

    @Test
    fun handlesReviewWithoutBoundingBoxes() {
        val e = event().copy(
            severityLevel = "Review",
            numFinalBoxes = 0,
            boxes = emptyList(),
            reason = "Classifier fired but detector found no matching box."
        )

        val a = AdvisoryParser.parse(
            e,
            "",
            100,
            0f
        )

        assertTrue(a.summary.contains("operator review"))
        assertTrue(a.recommendedAction.contains("Manually inspect"))
    }
}