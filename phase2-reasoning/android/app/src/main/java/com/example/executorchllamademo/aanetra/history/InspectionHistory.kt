package com.example.executorchllamademo.aanetra.history

import android.content.Context
import android.content.Intent
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.database.sqlite.SQLiteProgram
import android.graphics.Bitmap
import androidx.core.content.FileProvider
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import kotlin.math.max
import kotlin.math.roundToInt

/** [SqlDb] on the phone's SQLite (plain SQLiteOpenHelper; no Room / KSP needed). */
class AndroidSqlDb(context: Context) : SqlDb {

    private val helper = object : SQLiteOpenHelper(
        context.applicationContext, InspectionStore.DB_NAME, null, InspectionStore.DB_VERSION
    ) {
        override fun onCreate(db: SQLiteDatabase) {
            InspectionStore.SCHEMA.forEach(db::execSQL)
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit
    }

    private fun bind(p: SQLiteProgram, args: List<Any?>) {
        args.forEachIndexed { i, v ->
            val k = i + 1
            when (v) {
                null -> p.bindNull(k)
                is Boolean -> p.bindLong(k, if (v) 1L else 0L)
                is Int -> p.bindLong(k, v.toLong())
                is Long -> p.bindLong(k, v)
                is Float -> p.bindDouble(k, v.toDouble())
                is Double -> p.bindDouble(k, v)
                is String -> p.bindString(k, v)
                else -> throw IllegalArgumentException("unsupported SQL argument ${v.javaClass}")
            }
        }
    }

    override fun execute(sql: String, args: List<Any?>) {
        helper.writableDatabase.compileStatement(sql).use { bind(it, args); it.execute() }
    }

    override fun insert(sql: String, args: List<Any?>): Long =
        helper.writableDatabase.compileStatement(sql).use { bind(it, args); it.executeInsert() }

    override fun update(sql: String, args: List<Any?>): Int =
        helper.writableDatabase.compileStatement(sql).use { bind(it, args); it.executeUpdateDelete() }

    override fun query(sql: String, args: List<Any?>): List<Map<String, Any?>> {
        val strArgs = args.map { it?.toString() }.toTypedArray()
        return helper.readableDatabase.rawQuery(sql, strArgs).use { c ->
            val out = ArrayList<Map<String, Any?>>(c.count)
            while (c.moveToNext()) {
                val row = HashMap<String, Any?>(c.columnCount)
                for (i in 0 until c.columnCount) {
                    row[c.getColumnName(i)] = when (c.getType(i)) {
                        Cursor.FIELD_TYPE_NULL -> null
                        Cursor.FIELD_TYPE_INTEGER -> c.getLong(i)
                        Cursor.FIELD_TYPE_FLOAT -> c.getDouble(i)
                        else -> c.getString(i)
                    }
                }
                out.add(row)
            }
            out
        }
    }
}

/**
 * Inspection log on the phone: one SQLite row per inspection plus a small JPEG thumbnail in
 * filesDir/inspection_thumbs/. CSV export goes to cacheDir/exports/ and is shared through the
 * app's FileProvider (no network). Call from a background thread.
 */
class InspectionHistory(context: Context) {

    private val appContext = context.applicationContext
    private val store = InspectionStore(AndroidSqlDb(appContext))
    private val thumbsDir = File(appContext.filesDir, THUMBS_DIR)
    private val exportDir = File(appContext.cacheDir, EXPORT_DIR)

    /** Save the vision result; returns the row id. */
    fun save(record: InspectionRecord, image: Bitmap?): Long {
        val thumb = image?.let { saveThumbnail(it, record.timestampMs) }
        return store.insert(record.copy(thumbnailPath = thumb?.absolutePath))
    }

    fun updateAdvice(id: Long, text: String, adviceMs: Long, cached: Boolean): Boolean =
        store.updateAdvice(id, text, adviceMs, cached)

    fun all(): List<InspectionRecord> = store.all()

    fun get(id: Long): InspectionRecord? = store.get(id)

    /** Delete every row and every thumbnail. */
    fun deleteAll(): Int {
        val n = store.deleteAll()
        thumbsDir.listFiles()?.forEach { it.delete() }
        return n
    }

    /** Write all rows to a fresh CSV in cacheDir/exports/ (older exports are removed). */
    fun exportCsv(): File {
        exportDir.mkdirs()
        exportDir.listFiles()?.forEach { it.delete() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(exportDir, "aanetra_inspections_$stamp.csv")
        InspectionCsv.writeTo(file, store.all())
        return file
    }

    /** Android share sheet for an exported CSV (read permission granted to the chosen app only). */
    fun shareIntent(file: File): Intent {
        val uri = FileProvider.getUriForFile(appContext, "${appContext.packageName}.fileprovider", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/csv"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, "AANetra inspection log")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(send, "Export inspection log").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }

    private fun saveThumbnail(image: Bitmap, timestampMs: Long): File? = try {
        thumbsDir.mkdirs()
        val (w, h) = thumbnailSize(image.width, image.height)
        val scaled = if (w == image.width && h == image.height) image
        else Bitmap.createScaledBitmap(image, w, h, true)
        val file = File(thumbsDir, "thumb_${timestampMs}_${UUID.randomUUID().toString().take(8)}.jpg")
        file.outputStream().use { scaled.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        if (scaled !== image) scaled.recycle()
        file
    } catch (e: Exception) {
        null   // a missing thumbnail must never lose the inspection row
    }

    companion object {
        const val THUMBS_DIR = "inspection_thumbs"
        const val EXPORT_DIR = "exports"
        const val THUMB_MAX_SIDE = 256

        /** Size with the longest side at most [THUMB_MAX_SIDE] (never enlarged). */
        fun thumbnailSize(w: Int, h: Int, maxSide: Int = THUMB_MAX_SIDE): Pair<Int, Int> {
            val longest = max(w, h)
            if (longest <= maxSide) return w to h
            val s = maxSide.toDouble() / longest
            return max(1, (w * s).roundToInt()) to max(1, (h * s).roundToInt())
        }
    }
}
