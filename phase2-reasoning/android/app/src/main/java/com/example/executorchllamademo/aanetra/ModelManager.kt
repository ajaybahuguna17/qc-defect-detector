package com.example.executorchllamademo.aanetra

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.StatFs
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.coroutines.coroutineContext

/**
 * Finds, downloads (once), verifies and loads the Llama model.
 *
 * This is the ONLY code in the app that uses the network. After the two model files
 * are on the device, everything (vision + Llama) runs fully offline.
 */
object ModelManager {

    private const val TAG = "ModelManager"
    private const val PREFS = "aanetra_model_manager"
    private const val DEV_DIR = "/data/local/tmp/llama"
    private const val BUFFER = 1 shl 20

    data class ModelFile(
        val name: String,
        val size: Long,
        val sha256: String,
        val url: String
    )

    val MODEL = ModelFile(
        name = "llama3_2.pte",
        size = 802_247_040L,
        sha256 = "4064C629BE703F481498EE48E5BB5DE329405D7C4122194BA5B45109C97BB74C",
        url = "https://github.com/ajaybahuguna17/aanetra-models/releases/download/v1.0-llama/llama3_2.pte"
    )

    val TOKENIZER = ModelFile(
        name = "tokenizer.model",
        size = 2_183_982L,
        sha256 = "82E9D31979E92AB929CD544440F129D9ECD797B69E327F80F17E1C50D5551B55",
        url = "https://github.com/ajaybahuguna17/aanetra-models/releases/download/v1.0-llama/tokenizer.model"
    )

    private val FILES = listOf(MODEL, TOKENIZER)
    val TOTAL_BYTES: Long = FILES.sumOf { it.size }

    sealed interface State {
        data object Idle : State
        data object Checking : State
        /** Waiting for the operator to accept a download over mobile data. */
        data class NeedsMobileDataConsent(val bytes: Long) : State
        data class Downloading(val fileName: String, val doneBytes: Long, val totalBytes: Long) : State
        data class Verifying(val fileName: String) : State
        data object Loading : State
        data class Ready(val source: String, val loadMs: Long) : State
        data class Error(val message: String) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    /**
     * Single thread that owns the LlmModule: load and every generate() call run here.
     * Pass it to LlamaAdvisor as its generation executor.
     */
    val llmExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "aanetra-llm").apply { isDaemon = true }
    }
    private val llmDispatcher = llmExecutor.asCoroutineDispatcher()

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var job: Job? = null

    @Volatile
    var module: LlmModule? = null
        private set

    fun modelDir(context: Context): File {
        val dir = context.getExternalFilesDir(null)?.let { File(it, "llm") }
            ?: File(context.filesDir, "llm")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    /** Starts (or resumes) the setup. Safe to call repeatedly. */
    @Synchronized
    fun start(context: Context, allowMobileData: Boolean = false) {
        val current = _state.value
        if (current is State.Ready) return
        if (job?.isActive == true) return
        val appContext = context.applicationContext
        job = scope.launch {
            try {
                setup(appContext, allowMobileData)
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Throwable) {
                Log.e(TAG, "Model setup failed", e)
                _state.value = State.Error(e.message ?: e.javaClass.simpleName)
            }
        }
    }

    /** Operator pressed Cancel on the mobile-data dialog. */
    fun declineMobileData() {
        _state.value = State.Error(
            "Download cancelled. Connect to Wi-Fi and press Retry " +
                "(765 MB one-time download)."
        )
    }

    private suspend fun setup(context: Context, allowMobileData: Boolean) {
        _state.value = State.Checking
        val dir = modelDir(context)

        // (a) Files already in the app's model directory.
        if (FILES.all { File(dir, it.name).length() == it.size }) {
            val allOk = FILES.all { ensureVerified(context, File(dir, it.name), it) }
            if (allOk) {
                load(File(dir, MODEL.name), File(dir, TOKENIZER.name), "app storage")
                return
            }
            // A file failed verification and was deleted; fall through to download.
        }

        // (b) Developer path pushed with adb.
        val devModel = File(DEV_DIR, MODEL.name)
        val devTokenizer = File(DEV_DIR, TOKENIZER.name)
        if (devModel.canRead() && devTokenizer.canRead() &&
            devModel.length() > 0 && devTokenizer.length() > 0
        ) {
            load(devModel, devTokenizer, DEV_DIR)
            return
        }

        // (c) One-time download.
        val missing = FILES.filter { File(dir, it.name).length() != it.size }
        val missingBytes = missing.sumOf { it.size - partFile(dir, it).length().coerceAtMost(it.size) }

        when (networkType(context)) {
            NetworkType.NONE -> throw IOException(
                "No internet connection. The Llama model (765 MB) must be downloaded once. " +
                    "Connect to Wi-Fi and press Retry."
            )
            NetworkType.METERED -> if (!allowMobileData) {
                _state.value = State.NeedsMobileDataConsent(missingBytes)
                return
            }
            NetworkType.UNMETERED -> Unit
        }

        val free = StatFs(dir.absolutePath).availableBytes
        if (free < missingBytes + 50L * 1024 * 1024) {
            throw IOException(
                "Not enough storage: need ${mb(missingBytes)} MB free, " +
                    "only ${mb(free)} MB available."
            )
        }

        var alreadyDone = FILES.filter { it !in missing }.sumOf { it.size }
        for (file in missing) {
            val target = File(dir, file.name)
            val part = partFile(dir, file)
            download(file, part, alreadyDone)
            _state.value = State.Verifying(file.name)
            if (part.length() != file.size) {
                part.delete()
                throw IOException(
                    "${file.name}: size mismatch (${part.length()} of ${file.size} bytes). Press Retry."
                )
            }
            val hash = sha256(part)
            if (!hash.equals(file.sha256, ignoreCase = true)) {
                part.delete()
                throw IOException("${file.name}: SHA-256 check failed, file deleted. Press Retry.")
            }
            target.delete()
            if (!part.renameTo(target)) {
                throw IOException("Could not save ${file.name}.")
            }
            markVerified(context, target)
            alreadyDone += file.size
        }

        load(File(dir, MODEL.name), File(dir, TOKENIZER.name), "app storage")
    }

    /** Hash once, then trust the file while its size + lastModified are unchanged. */
    private suspend fun ensureVerified(context: Context, file: File, spec: ModelFile): Boolean {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (prefs.getString(file.name, null) == verifiedKey(file)) return true

        _state.value = State.Verifying(file.name)
        val hash = sha256(file)
        return if (hash.equals(spec.sha256, ignoreCase = true)) {
            markVerified(context, file)
            true
        } else {
            Log.w(TAG, "${file.name} failed SHA-256 check; deleting")
            file.delete()
            prefs.edit().remove(file.name).apply()
            false
        }
    }

    private fun markVerified(context: Context, file: File) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(file.name, verifiedKey(file)).apply()
    }

    private fun verifiedKey(file: File) = "${file.length()}:${file.lastModified()}"

    private fun partFile(dir: File, spec: ModelFile) = File(dir, spec.name + ".part")

    private suspend fun download(spec: ModelFile, part: File, alreadyDoneBytes: Long) {
        var attemptsLeft = 2
        while (true) {
            coroutineContext.ensureActive()
            var existing = part.length()
            if (existing > spec.size) {
                part.delete()
                existing = 0L
            }
            if (existing == spec.size) return

            val conn = openFollowingRedirects(spec.url, existing)
            try {
                val code = conn.responseCode
                val append = when {
                    code == HttpURLConnection.HTTP_PARTIAL && existing > 0 -> true
                    code == HttpURLConnection.HTTP_OK -> false
                    code == 416 -> {
                        // Range not satisfiable: our .part is unusable, start again.
                        part.delete()
                        if (attemptsLeft-- > 0) continue
                        throw IOException("${spec.name}: server rejected resume (HTTP 416).")
                    }
                    else -> throw IOException("${spec.name}: download failed (HTTP $code).")
                }
                if (!append) existing = 0L

                var done = existing
                var lastUpdate = 0L
                _state.value = State.Downloading(spec.name, alreadyDoneBytes + done, TOTAL_BYTES)

                conn.inputStream.use { input ->
                    FileOutputStream(part, append).use { out ->
                        val buf = ByteArray(BUFFER / 4)
                        while (true) {
                            coroutineContext.ensureActive()
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            done += n
                            val now = System.currentTimeMillis()
                            if (now - lastUpdate > 250) {
                                lastUpdate = now
                                _state.value = State.Downloading(
                                    spec.name, alreadyDoneBytes + done, TOTAL_BYTES
                                )
                            }
                        }
                    }
                }
                _state.value = State.Downloading(spec.name, alreadyDoneBytes + done, TOTAL_BYTES)
                if (done < spec.size) {
                    throw IOException(
                        "${spec.name}: connection closed early (${mb(done)} of ${mb(spec.size)} MB). " +
                            "Press Retry to resume."
                    )
                }
                return
            } catch (e: IOException) {
                if (e.message?.startsWith(spec.name) == true) throw e
                throw IOException(
                    "Download interrupted: ${e.message ?: e.javaClass.simpleName}. " +
                        "Press Retry to resume.", e
                )
            } finally {
                conn.disconnect()
            }
        }
    }

    /**
     * GitHub release assets redirect to objects.githubusercontent.com.
     * Redirects are followed manually so the Range header is sent on every hop.
     */
    private fun openFollowingRedirects(url: String, rangeStart: Long): HttpURLConnection {
        var current = URL(url)
        repeat(8) {
            val conn = (current.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 20_000
                readTimeout = 30_000
                setRequestProperty("User-Agent", "AANetra-Android")
                setRequestProperty("Accept-Encoding", "identity")
                if (rangeStart > 0) setRequestProperty("Range", "bytes=$rangeStart-")
            }
            val code = conn.responseCode
            if (code in 300..399) {
                val location = conn.getHeaderField("Location")
                conn.disconnect()
                if (location.isNullOrBlank()) throw IOException("Redirect without location (HTTP $code).")
                current = URL(current, location)
                if (current.protocol != "https") throw IOException("Refusing non-HTTPS redirect.")
            } else {
                return conn
            }
        }
        throw IOException("Too many redirects.")
    }

    private suspend fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buf = ByteArray(BUFFER)
            while (true) {
                coroutineContext.ensureActive()
                val n = input.read(buf)
                if (n < 0) break
                digest.update(buf, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02X".format(Locale.US, it) }
    }

    private suspend fun load(model: File, tokenizer: File, source: String) {
        _state.value = State.Loading
        val start = System.currentTimeMillis()
        val loaded = withContext(llmDispatcher) {
            module?.let { return@withContext it }
            val m = LlmModule(model.absolutePath, tokenizer.absolutePath, 0f)
            try {
                m.load()
            } catch (e: Throwable) {
                runCatching { m.close() }
                throw IllegalStateException("Llama failed to load: ${e.message}", e)
            }
            m
        }
        module = loaded
        _state.value = State.Ready(source, System.currentTimeMillis() - start)
    }

    private enum class NetworkType { NONE, METERED, UNMETERED }

    private fun networkType(context: Context): NetworkType {
        val cm = context.getSystemService(ConnectivityManager::class.java) ?: return NetworkType.NONE
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return NetworkType.NONE
        if (!caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)) return NetworkType.NONE
        val unmetered = caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) ||
            caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
            caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)
        val cellular = caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR)
        return if (cellular || !unmetered) NetworkType.METERED else NetworkType.UNMETERED
    }

    fun mb(bytes: Long): Long = bytes / (1024L * 1024L)
}
