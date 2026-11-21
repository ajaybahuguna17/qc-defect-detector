package com.example.executorchllamademo.aanetra.history

/**
 * Minimal SQL access used by [InspectionStore]. On the phone it is [AndroidSqlDb]
 * (SQLiteOpenHelper); the JVM unit tests run the same SQL on SQLite through JDBC.
 * Positional `?` arguments: null, Long/Int, Double/Float, String, Boolean (stored 0/1).
 */
interface SqlDb {
    fun execute(sql: String, args: List<Any?> = emptyList())
    /** INSERT; returns the new row id. */
    fun insert(sql: String, args: List<Any?>): Long
    /** UPDATE / DELETE; returns the number of rows changed. */
    fun update(sql: String, args: List<Any?>): Int
    /** SELECT; each row maps column name -> null, Long, Double or String. */
    fun query(sql: String, args: List<Any?> = emptyList()): List<Map<String, Any?>>
}

/** The `inspections` table: insert after vision, update when advice arrives, read newest first. */
class InspectionStore(private val db: SqlDb) {

    fun createSchema() {
        SCHEMA.forEach { db.execute(it) }
    }

    fun insert(r: InspectionRecord): Long = db.insert(
        "INSERT INTO $TABLE (timestamp_ms, source, thumbnail_path, defect_type, severity, decision, " +
            "decided_by, classifier_confidence, defects_found, vision_ms, advice_text, advice_ms, " +
            "advice_cached, gate_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        listOf(
            r.timestampMs, r.source, r.thumbnailPath, r.defectType, r.severity, r.decision,
            r.decidedBy, r.classifierConfidence?.toDouble(), r.defectsFound, r.visionMs,
            r.adviceText, r.adviceMs, r.adviceCached, r.gateReason
        )
    )

    /** Fill in the advice of row [id]; false when the row no longer exists (e.g. deleted). */
    fun updateAdvice(id: Long, text: String, adviceMs: Long, cached: Boolean): Boolean =
        db.update(
            "UPDATE $TABLE SET advice_text = ?, advice_ms = ?, advice_cached = ? WHERE id = ?",
            listOf(text, adviceMs, cached, id)
        ) == 1

    /** All rows, newest first (ties: higher id first). */
    fun all(): List<InspectionRecord> =
        db.query("SELECT * FROM $TABLE ORDER BY timestamp_ms DESC, id DESC").map(::toRecord)

    fun get(id: Long): InspectionRecord? =
        db.query("SELECT * FROM $TABLE WHERE id = ?", listOf(id)).firstOrNull()?.let(::toRecord)

    fun deleteAll(): Int = db.update("DELETE FROM $TABLE", emptyList())

    fun count(): Int = (db.query("SELECT COUNT(*) AS n FROM $TABLE").single()["n"] as Number).toInt()

    private fun toRecord(row: Map<String, Any?>): InspectionRecord {
        fun long(k: String) = (row[k] as Number?)?.toLong()
        fun str(k: String) = row[k] as String?
        return InspectionRecord(
            id = long("id")!!,
            timestampMs = long("timestamp_ms")!!,
            source = str("source")!!,
            thumbnailPath = str("thumbnail_path"),
            defectType = str("defect_type")!!,
            severity = str("severity"),
            decision = str("decision")!!,
            decidedBy = str("decided_by")!!,
            classifierConfidence = (row["classifier_confidence"] as Number?)?.toFloat(),
            defectsFound = long("defects_found")!!.toInt(),
            visionMs = long("vision_ms")!!,
            adviceText = str("advice_text"),
            adviceMs = long("advice_ms"),
            adviceCached = long("advice_cached")?.let { it != 0L },
            gateReason = str("gate_reason")
        )
    }

    companion object {
        const val TABLE = "inspections"
        const val DB_NAME = "aanetra_inspections.db"
        const val DB_VERSION = 1

        val SCHEMA = listOf(
            """
            CREATE TABLE IF NOT EXISTS $TABLE (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                source TEXT NOT NULL,
                thumbnail_path TEXT,
                defect_type TEXT NOT NULL,
                severity TEXT,
                decision TEXT NOT NULL,
                decided_by TEXT NOT NULL,
                classifier_confidence REAL,
                defects_found INTEGER NOT NULL,
                vision_ms INTEGER NOT NULL,
                advice_text TEXT,
                advice_ms INTEGER,
                advice_cached INTEGER,
                gate_reason TEXT
            )
            """.trimIndent(),
            "CREATE INDEX IF NOT EXISTS idx_${TABLE}_time ON $TABLE (timestamp_ms)"
        )
    }
}
