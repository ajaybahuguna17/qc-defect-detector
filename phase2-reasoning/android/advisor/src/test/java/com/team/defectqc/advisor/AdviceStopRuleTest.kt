package com.team.defectqc.advisor

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AdviceStopRuleTest {

    private fun feed(vararg chunks: String): Pair<AdviceStopRule, Int?> {
        val r = AdviceStopRule()
        var stoppedAt: Int? = null
        chunks.forEachIndexed { i, c -> if (r.onChunk(c) && stoppedAt == null) stoppedAt = i }
        return r to stoppedAt
    }

    @Test
    fun stopsOnNewlineAfterActionText() {
        val (r, at) = feed("SUMMARY: Scratch found.\n", "CAUSE: Roller wear.\n", "ACTION: Inspect", " rollers.", "\n", "Extra")
        assertEquals(4, at)                       // the "\n" chunk
        assertTrue(r.stopped)
        assertEquals("SUMMARY: Scratch found.\nCAUSE: Roller wear.\nACTION: Inspect rollers.\n", r.text)
    }

    @Test
    fun stopsOnEotAfterActionText() {
        val (_, at) = feed("SUMMARY: a\nCAUSE: b\nACTION: check", " the part<|eot_id|>")
        assertEquals(1, at)
    }

    @Test
    fun doesNotStopMidActionLine() {
        val (r, at) = feed("SUMMARY: a\n", "CAUSE: b\n", "ACTION: Inspect the")
        assertNull(at)
        assertFalse(r.stopped)
    }

    @Test
    fun emptyActionLineDoesNotStop() {
        assertFalse(AdviceStopRule.isComplete("SUMMARY: a\nACTION:\n"))
        assertFalse(AdviceStopRule.isComplete("ACTION:   \n"))
        assertTrue(AdviceStopRule.isComplete("ACTION:\nACTION: do it\n"))
    }

    @Test
    fun noActionNeverStops() {
        val (r, at) = feed("SUMMARY: a\n", "CAUSE: b\n", "Some rambling text\n", "<|eot_id|>")
        assertNull(at)
        assertFalse(r.stopped)
    }

    @Test
    fun matchesParserFormat() {
        // same as AdvisoryParser: any case, leading spaces, spaces before ':'
        assertTrue(AdviceStopRule.isComplete("  action : Do it\n"))
        assertTrue(AdviceStopRule.isComplete("Action: Do it\r\n"))
        // not at line start -> parser would not read it either
        assertFalse(AdviceStopRule.isComplete("The ACTION: is unclear\n"))
        assertFalse(AdviceStopRule.isComplete("**ACTION:** Do it\n"))
    }

    @Test
    fun textAfterEotIsIgnored() {
        assertFalse(AdviceStopRule.isComplete("SUMMARY: a<|eot_id|>ACTION: x\n"))
    }

    @Test
    fun onlyReturnsTrueOnce() {
        val r = AdviceStopRule()
        assertTrue(r.onChunk("ACTION: x\n"))
        assertFalse(r.onChunk("more\n"))
        assertEquals("ACTION: x\n", r.text)
    }

    @Test
    fun parsedAdviceFromStoppedTextHasAllFields() {
        val (r, _) = feed("SUMMARY: s1\n", "CAUSE: c1\n", "ACTION: a1\n")
        val event = com.team.defectqc.core.DetectionEvent(
            id = "e", timestampEpochMs = 0L, imageName = "x.jpg", detector = "fp32", alert = true,
            finalClass = "scratches", decidedBy = "classifier", classifierClass = "scratches",
            classifierConfidence = 0.9f, numBoxes = 1, numFinalBoxes = 1, largestAreaPct = 5f,
            severityLevel = "Low", reason = "r",
            boxes = listOf(com.team.defectqc.core.DefectBox("scratches", 0.9f, listOf(0f, 0f, 1f, 1f)))
        )
        val a = AdvisoryParser.parse(event, r.text, 1L, 1f)
        assertEquals("s1", a.summary)
        assertEquals("c1", a.likelyCause)
        assertEquals("a1", a.recommendedAction)
    }

    @Test
    fun statsParsing() {
        val s = GenerationStats.parse(
            """{"prompt_tokens":182,"generated_tokens":64,"model_load_start_ms":0,"model_load_end_ms":0,""" +
                """"inference_start_ms":1000,"token_encode_end_ms":1010,"model_execution_start_ms":1010,""" +
                """"model_execution_end_ms":5000,"inference_end_ms":5200,"prompt_eval_end_ms":2200,""" +
                """"first_token_ms":2200,"aggregate_sampling_time_ms":30,"SCALING_FACTOR_UNITS_PER_SECOND":1000}"""
        )
        assertEquals(182, s.promptTokens)
        assertEquals(64, s.generatedTokens)
        assertEquals(1200L, s.ttftMs)
        assertEquals(64f * 1000f / 3000f, s.tokensPerSecond, 1e-3f)
    }

    @Test
    fun statsParsingMissingFields() {
        val s = GenerationStats.parse("")
        assertNull(s.promptTokens)
        assertNull(s.ttftMs)
        assertEquals(0f, s.tokensPerSecond)
    }
}
