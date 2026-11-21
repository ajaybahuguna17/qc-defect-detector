package com.example.executorchllamademo.aanetra.history

import com.team.defectqc.core.DetectionEvent
import com.team.defectqc.core.QualityDecision
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.sql.Connection
import java.sql.DriverManager
import java.sql.Statement
import java.sql.Types

/** [SqlDb] on an in-memory SQLite database through JDBC: the same SQL the phone runs. */
class JdbcSqlDb : SqlDb, AutoCloseable {
    private val conn: Connection = DriverManager.getConnection("jdbc:sqlite::memory:")

    private fun prepare(sql: String, args: List<Any?>, keys: Boolean = false) =
        conn.prepareStatement(sql, if (keys) Statement.RETURN_GENERATED_KEYS else Statement.NO_GENERATED_KEYS).apply {
            args.forEachIndexed { i, v ->
                val k = i + 1
                when (v) {
                    null -> setNull(k, Types.NULL)
                    is Boolean -> setLong(k, if (v) 1L else 0L)
                    is Int -> setLong(k, v.toLong())
                    is Long -> setLong(k, v)
                    is Float -> setDouble(k, v.toDouble())
                    is Double -> setDouble(k, v)
                    is String -> setString(k, v)
                    else -> throw IllegalArgumentException("unsupported ${v.javaClass}")
                }
            }
        }

    override fun execute(sql: String, args: List<Any?>) {
        prepare(sql, args).use { it.execute() }
    }

    override fun insert(sql: String, args: List<Any?>): Long = prepare(sql, args).use { st ->
        st.executeUpdate()
        conn.createStatement().use { s -> s.executeQuery("SELECT last_insert_rowid()").use { it.next(); it.getLong(1) } }
    }

    override fun update(sql: String, args: List<Any?>): Int = prepare(sql, args).use { it.executeUpdate() }

    override fun query(sql: String, args: List<Any?>): List<Map<String, Any?>> = prepare(sql, args).use { st ->
        st.executeQuery().use { rs ->
            val md = rs.metaData
            val out = mutableListOf<Map<String, Any?>>()
            while (rs.next()) {
                out += (1..md.columnCount).associate { i ->
                    md.getColumnLabel(i) to when (val v = rs.getObject(i)) {
                        is Int -> v.toLong()
                        is Float -> v.toDouble()
                        else -> v
                    }
                }
            }
            out
        }
    }

    override fun close() = conn.close()
}

class InspectionStoreTest {

    private lateinit var db: JdbcSqlDb
    private lateinit var store: InspectionStore

    @Before
    fun setUp() {
        db = JdbcSqlDb()
        store = InspectionStore(db)
        store.createSchema()
    }

    @After
    fun tearDown() = db.close()

    private fun record(ts: Long, defect: String = "scratches", decision: String = "REWORK", reason: String? = null) =
        InspectionRecord(
            timestampMs = ts, source = InspectionRecord.SOURCE_CAMERA, thumbnailPath = "/data/x/thumb_$ts.jpg",
            defectType = defect, severity = "Medium", decision = decision, decidedBy = "classifier",
            classifierConfidence = 0.875f, defectsFound = 2, visionMs = 123L, gateReason = reason
        )

    @Test
    fun insertAndReadBackAllFields() {
        val id = store.insert(record(1_700_000_000_000L, reason = "why"))
        val r = store.get(id)!!
        assertEquals(id, r.id)
        assertEquals(1_700_000_000_000L, r.timestampMs)
        assertEquals("camera", r.source)
        assertEquals("/data/x/thumb_1700000000000.jpg", r.thumbnailPath)
        assertEquals("scratches", r.defectType)
        assertEquals("Medium", r.severity)
        assertEquals("REWORK", r.decision)
        assertEquals("classifier", r.decidedBy)
        assertEquals(0.875f, r.classifierConfidence!!, 1e-6f)
        assertEquals(2, r.defectsFound)
        assertEquals(123L, r.visionMs)
        assertEquals("why", r.gateReason)
        assertNull(r.adviceText); assertNull(r.adviceMs); assertNull(r.adviceCached)
    }

    @Test
    fun nullsRoundTrip() {
        val id = store.insert(record(5L).copy(severity = null, classifierConfidence = null, thumbnailPath = null))
        val r = store.get(id)!!
        assertNull(r.severity); assertNull(r.classifierConfidence); assertNull(r.thumbnailPath)
    }

    @Test
    fun updateAdviceFillsRow() {
        val id = store.insert(record(10L))
        assertTrue(store.updateAdvice(id, "SUMMARY: a\nCAUSE: b\nACTION: c", 4200L, cached = true))
        val r = store.get(id)!!
        assertEquals("SUMMARY: a\nCAUSE: b\nACTION: c", r.adviceText)
        assertEquals(4200L, r.adviceMs)
        assertEquals(true, r.adviceCached)
        assertTrue(store.updateAdvice(id, "x", 1L, cached = false))
        assertEquals(false, store.get(id)!!.adviceCached)
    }

    @Test
    fun updateAdviceOnMissingRowReturnsFalse() {
        assertFalse(store.updateAdvice(999L, "x", 1L, false))
    }

    @Test
    fun allIsNewestFirst() {
        val a = store.insert(record(100L, "crazing"))
        val b = store.insert(record(300L, "patches"))
        val c = store.insert(record(200L, "inclusion"))
        val d = store.insert(record(300L, "scratches"))   // same time as b, later id -> first
        assertEquals(listOf(d, b, c, a), store.all().map { it.id })
        assertEquals(4, store.count())
    }

    @Test
    fun deleteAllEmptiesTable() {
        store.insert(record(1L)); store.insert(record(2L))
        assertEquals(2, store.deleteAll())
        assertEquals(0, store.count())
        assertTrue(store.all().isEmpty())
    }

    @Test
    fun schemaIsIdempotent() {
        store.createSchema()
        store.insert(record(1L))
        assertEquals(1, store.count())
    }

    @Test
    fun recordFromInspection() {
        val event = DetectionEvent(
            id = "e", timestampEpochMs = 42L, imageName = "x.jpg", detector = "fp32", alert = false,
            finalClass = "not a steel surface", decidedBy = "classifier", classifierClass = "not_steel",
            classifierConfidence = 0.99f, numBoxes = 0, numFinalBoxes = 0, largestAreaPct = null,
            severityLevel = null, reason = "classifier not_steel 0.990 >= 0.33", boxes = emptyList(),
            decision = "RETAKE"
        )
        val r = InspectionRecord.fromInspection(event, QualityDecision.decide(event), 77L, InspectionRecord.SOURCE_GALLERY, null)
        assertEquals("RETAKE", r.decision)
        assertEquals("classifier not_steel 0.990 >= 0.33", r.gateReason)
        assertEquals(42L, r.timestampMs)
        assertEquals("gallery", r.source)
        // a normal defect has no gate reason
        val ok = event.copy(alert = true, finalClass = "scratches", severityLevel = "High", decision = null, numBoxes = 1)
        val r2 = InspectionRecord.fromInspection(ok, QualityDecision.decide(ok), 1L, "camera", null)
        assertEquals("REJECT", r2.decision)
        assertNull(r2.gateReason)
        assertEquals(1, r2.defectsFound)
    }

    @Test
    fun thumbnailSizeKeepsAspectAndMax256() {
        assertEquals(256 to 192, InspectionHistory.thumbnailSize(4000, 3000))
        assertEquals(144 to 256, InspectionHistory.thumbnailSize(1080, 1920))
        assertEquals(200 to 200, InspectionHistory.thumbnailSize(200, 200))   // never enlarged
    }
}
