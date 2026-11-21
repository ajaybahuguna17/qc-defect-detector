package com.example.executorchllamademo.aanetra.history

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files
import java.time.ZoneId

class InspectionCsvTest {

    private val zone = ZoneId.of("Asia/Kolkata")

    private fun record(
        id: Long = 7L,
        advice: String? = "SUMMARY: ok\nCAUSE: roller, \"worn\"\nACTION: check",
        reason: String? = null
    ) = InspectionRecord(
        id = id, timestampMs = 1_763_712_000_000L, source = "camera", thumbnailPath = "/data/files/inspection_thumbs/t1.jpg",
        defectType = "scratches", severity = "High", decision = "REJECT + STOP LINE", decidedBy = "classifier",
        classifierConfidence = 0.98771f, defectsFound = 3, visionMs = 250L, adviceText = advice, adviceMs = 5400L,
        adviceCached = false, gateReason = reason
    )

    /** Minimal RFC 4180 parser for checking the output. */
    private fun parse(csv: String): List<List<String>> {
        val rows = mutableListOf<List<String>>()
        var row = mutableListOf<String>()
        val f = StringBuilder()
        var i = 0
        var quoted = false
        while (i < csv.length) {
            val c = csv[i]
            if (quoted) {
                if (c == '"' && i + 1 < csv.length && csv[i + 1] == '"') { f.append('"'); i++ }
                else if (c == '"') quoted = false
                else f.append(c)
            } else when (c) {
                '"' -> quoted = true
                ',' -> { row.add(f.toString()); f.clear() }
                '\r' -> {}
                '\n' -> { row.add(f.toString()); f.clear(); rows.add(row); row = mutableListOf() }
                else -> f.append(c)
            }
            i++
        }
        return rows
    }

    @Test
    fun headerAndOneRow() {
        val rows = parse(InspectionCsv.write(listOf(record()), zone))
        assertEquals(InspectionCsv.HEADER, rows[0])
        val r = rows[1].zip(rows[0]).associate { (v, k) -> k to v }
        assertEquals("7", r["id"])
        assertEquals("2025-11-21T13:30:00+05:30", r["inspected_at_local"])
        assertEquals("1763712000000", r["timestamp_ms"])
        assertEquals("REJECT + STOP LINE", r["decision"])
        assertEquals("0.9877", r["classifier_confidence"])
        assertEquals("3", r["defects_found"])
        assertEquals("5400", r["advice_ms"])
        assertEquals("no", r["advice_cached"])
        assertEquals("SUMMARY: ok\nCAUSE: roller, \"worn\"\nACTION: check", r["advice_text"])   // quoting round-trips
        assertEquals("t1.jpg", r["thumbnail_file"])                                          // file name only
        assertEquals(InspectionCsv.HEADER.size, rows[1].size)
    }

    @Test
    fun crlfLineEnds() {
        val csv = InspectionCsv.write(listOf(record(advice = null)), zone)
        assertTrue(csv.endsWith("\r\n"))
        assertEquals(2, csv.split("\r\n").filter { it.isNotEmpty() }.size)
    }

    @Test
    fun emptyLogIsHeaderOnly() {
        assertEquals(InspectionCsv.HEADER.joinToString(",") + "\r\n", InspectionCsv.write(emptyList(), zone))
    }

    @Test
    fun nullsAreEmpty() {
        val rows = parse(InspectionCsv.write(listOf(record(advice = null).copy(adviceMs = null, adviceCached = null,
            severity = null, classifierConfidence = null, thumbnailPath = null)), zone))
        val r = rows[1].zip(rows[0]).associate { (v, k) -> k to v }
        listOf("advice_text", "advice_ms", "advice_cached", "severity", "classifier_confidence", "thumbnail_file")
            .forEach { assertEquals(it, "", r[it]) }
    }

    @Test
    fun formulaInjectionIsNeutralised() {
        val rows = parse(InspectionCsv.write(listOf(record(advice = "=HYPERLINK(\"x\")", reason = "+1")), zone))
        val r = rows[1].zip(rows[0]).associate { (v, k) -> k to v }
        assertEquals("'=HYPERLINK(\"x\")", r["advice_text"])
        assertEquals("'+1", r["retake_hold_reason"])
    }

    @Test
    fun escapeRules() {
        assertEquals("plain", InspectionCsv.escape("plain"))
        assertEquals("\"a,b\"", InspectionCsv.escape("a,b"))
        assertEquals("\"say \"\"hi\"\"\"", InspectionCsv.escape("say \"hi\""))
        assertEquals("\"l1\nl2\"", InspectionCsv.escape("l1\nl2"))
    }

    @Test
    fun writeToFileHasBomAndUtf8() {
        val dir = Files.createTempDirectory("csv").toFile()
        try {
            val f = File(dir, "exports/x.csv")
            InspectionCsv.writeTo(f, listOf(record(advice = "Prüfen – ok")), zone)
            val bytes = f.readBytes()
            assertEquals(0xEF.toByte(), bytes[0]); assertEquals(0xBB.toByte(), bytes[1]); assertEquals(0xBF.toByte(), bytes[2])
            assertTrue(f.readText(Charsets.UTF_8).contains("Prüfen – ok"))
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun storeToCsvEndToEnd() {
        JdbcSqlDb().use { db ->
            val store = InspectionStore(db).also { it.createSchema() }
            val a = store.insert(record(id = 0L).copy(timestampMs = 1000L))
            val b = store.insert(record(id = 0L).copy(timestampMs = 2000L, decision = "RETAKE", gateReason = "too dark"))
            store.updateAdvice(a, "SUMMARY: s\nCAUSE: c\nACTION: a", 100L, true)
            val rows = parse(InspectionCsv.write(store.all(), zone))
            assertEquals(3, rows.size)
            assertEquals(b.toString(), rows[1][0])     // newest first
            assertEquals("RETAKE", rows[1][6])
            assertEquals("too dark", rows[1][13])
            assertEquals(a.toString(), rows[2][0])
            assertEquals("yes", rows[2][12])
        }
    }
}
