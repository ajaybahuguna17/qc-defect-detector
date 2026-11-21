package com.example.executorchllamademo.aanetra.history

import java.io.File
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * CSV export of the inspection log (RFC 4180: comma separated, CRLF line ends, fields with
 * comma / quote / line break quoted, quotes doubled). UTF-8 with a BOM so spreadsheet apps read
 * the text correctly. Text fields that start with = + - @ (or a tab / CR) get a leading ' so a
 * spreadsheet never runs them as a formula.
 */
object InspectionCsv {

    val HEADER = listOf(
        "id", "inspected_at_local", "timestamp_ms", "source", "defect_type", "severity", "decision",
        "decided_by", "classifier_confidence", "defects_found", "vision_ms", "advice_ms",
        "advice_cached", "retake_hold_reason", "advice_text", "thumbnail_file"
    )

    private val ISO_LOCAL = DateTimeFormatter.ISO_OFFSET_DATE_TIME

    fun write(records: List<InspectionRecord>, zone: ZoneId = ZoneId.systemDefault()): String {
        val sb = StringBuilder()
        sb.append(HEADER.joinToString(",")).append("\r\n")
        for (r in records) {
            val row = listOf(
                r.id.toString(),
                ISO_LOCAL.format(Instant.ofEpochMilli(r.timestampMs).atZone(zone).withNano(0)),
                r.timestampMs.toString(),
                text(r.source),
                text(r.defectType),
                text(r.severity),
                text(r.decision),
                text(r.decidedBy),
                r.classifierConfidence?.let { String.format(Locale.US, "%.4f", it) } ?: "",
                r.defectsFound.toString(),
                r.visionMs.toString(),
                r.adviceMs?.toString() ?: "",
                when (r.adviceCached) { true -> "yes"; false -> "no"; null -> "" },
                text(r.gateReason),
                text(r.adviceText),
                text(r.thumbnailPath?.let { File(it).name })
            )
            sb.append(row.joinToString(",") { escape(it) }).append("\r\n")
        }
        return sb.toString()
    }

    fun writeTo(file: File, records: List<InspectionRecord>, zone: ZoneId = ZoneId.systemDefault()) {
        file.parentFile?.mkdirs()
        file.writeText("﻿" + write(records, zone), Charsets.UTF_8)
    }

    /** Formula guard for free text (numbers are written by the code and never guarded). */
    internal fun text(v: String?): String {
        if (v.isNullOrEmpty()) return ""
        return if (v[0] in "=+-@\t\r") "'$v" else v
    }

    internal fun escape(v: String): String =
        if (v.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) "\"" + v.replace("\"", "\"\"") + "\"" else v
}
