package com.example.executorchllamademo.aanetra

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.executorchllamademo.ETLogging
import com.team.defectqc.advisor.LlamaAdvisor
import com.team.defectqc.core.Advisory
import com.team.defectqc.core.DetectionEvent
import com.team.defectqc.core.QualityDecision
import com.team.defectqc.vision.DefectVisionPipeline
import com.team.defectqc.vision.VisionResultMapper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class InspectionViewModel(app: Application) : AndroidViewModel(app) {

    sealed interface VisionStatus {
        data object Loading : VisionStatus
        data object Ready : VisionStatus
        data class Error(val message: String) : VisionStatus
    }

    sealed interface AdviceState {
        data object None : AdviceState
        data object NoDefect : AdviceState
        data class NotReady(val reason: String) : AdviceState
        data class Generating(val startedAtMs: Long) : AdviceState
        data class Done(val advisory: Advisory) : AdviceState
        /** Details are logged (Log.e + Developer settings logs), never shown here. */
        data object Failed : AdviceState
    }

    data class InspectionResult(
        val bitmap: Bitmap,
        val event: DetectionEvent,
        val decision: QualityDecision,
        val visionMs: Long
    )

    private val _visionStatus = MutableStateFlow<VisionStatus>(VisionStatus.Loading)
    val visionStatus: StateFlow<VisionStatus> = _visionStatus.asStateFlow()

    private val _busy = MutableStateFlow(false)
    val busy: StateFlow<Boolean> = _busy.asStateFlow()

    private val _result = MutableStateFlow<InspectionResult?>(null)
    val result: StateFlow<InspectionResult?> = _result.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    private val _advice = MutableStateFlow<AdviceState>(AdviceState.None)
    val advice: StateFlow<AdviceState> = _advice.asStateFlow()

    val modelState: StateFlow<ModelManager.State> = ModelManager.state

    @Volatile
    private var pipeline: DefectVisionPipeline? = null

    private val advisor = LlamaAdvisor(
        modelProvider = { ModelManager.module },
        generationExecutor = ModelManager.llmExecutor,
        // Total length (prompt + answer) = the .pte max context. The prompt alone is ~200 tokens.
        sequenceLength = 512
    )

    private var adviceJob: Job? = null

    init {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                pipeline = DefectVisionPipeline(getApplication())
                _visionStatus.value = VisionStatus.Ready
            } catch (e: Throwable) {
                Log.e(TAG, "Vision pipeline failed to load", e)
                _visionStatus.value = VisionStatus.Error(e.message ?: e.javaClass.simpleName)
            }
        }

        ModelManager.start(app)

        // If Llama becomes ready while a defect result is waiting, generate advice then.
        viewModelScope.launch {
            ModelManager.state.collect { state ->
                val current = _result.value ?: return@collect
                when (state) {
                    is ModelManager.State.Ready ->
                        if (_advice.value is AdviceState.NotReady) requestAdvice(current.event)
                    else ->
                        if (_advice.value is AdviceState.NotReady) {
                            _advice.value = AdviceState.NotReady(notReadyReason(state))
                        }
                }
            }
        }
    }

    fun retryModel() = ModelManager.start(getApplication())

    fun acceptMobileData() = ModelManager.start(getApplication(), allowMobileData = true)

    fun declineMobileData() = ModelManager.declineMobileData()

    fun retryAdvice() {
        val current = _result.value ?: return
        if (current.event.alert && _advice.value is AdviceState.Failed) requestAdvice(current.event)
    }

    fun inspect(uri: Uri) {
        if (_busy.value) return
        val vision = pipeline
        if (vision == null) {
            _error.value = "Vision models are not ready yet."
            return
        }
        _busy.value = true
        _error.value = null
        adviceJob?.cancel()

        viewModelScope.launch {
            try {
                val inspection = withContext(Dispatchers.Default) {
                    val bitmap = decodeBitmap(uri)
                    val start = SystemClock.elapsedRealtime()
                    val visionResult = vision.analyze(bitmap)
                    val visionMs = SystemClock.elapsedRealtime() - start
                    val event = VisionResultMapper.toDetectionEvent(
                        result = visionResult,
                        imageName = uri.lastPathSegment ?: "inspection_image"
                    )
                    InspectionResult(bitmap, event, QualityDecision.decide(event), visionMs)
                }
                _result.value = inspection
                if (inspection.event.alert) {
                    requestAdvice(inspection.event)
                } else {
                    _advice.value = AdviceState.NoDefect
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Inspection failed", e)
                _error.value = "Inspection failed: ${e.message ?: e.javaClass.simpleName}"
            } finally {
                _busy.value = false
            }
        }
    }

    private fun requestAdvice(event: DetectionEvent) {
        val state = ModelManager.state.value
        if (state !is ModelManager.State.Ready || ModelManager.module == null) {
            _advice.value = AdviceState.NotReady(notReadyReason(state))
            return
        }
        adviceJob?.cancel()
        _advice.value = AdviceState.Generating(SystemClock.elapsedRealtime())
        adviceJob = viewModelScope.launch {
            try {
                val advisory = advisor.advise(event)
                if (_result.value?.event?.id == event.id) {
                    _advice.value = AdviceState.Done(advisory)
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Throwable) {
                Log.e(TAG, "Advice failed", e)
                ETLogging.getInstance().log(
                    "AANetra advice failed: ${e.javaClass.simpleName}: ${e.message}\n" +
                        Log.getStackTraceString(e)
                )
                if (_result.value?.event?.id == event.id) {
                    _advice.value = AdviceState.Failed
                }
            }
        }
    }

    private fun notReadyReason(state: ModelManager.State): String = when (state) {
        is ModelManager.State.Downloading ->
            "Llama model is still downloading (${percent(state.doneBytes, state.totalBytes)}%)."
        is ModelManager.State.Verifying -> "Llama model is being verified."
        ModelManager.State.Loading -> "Llama model is loading."
        ModelManager.State.Checking, ModelManager.State.Idle -> "Llama model is being prepared."
        is ModelManager.State.NeedsMobileDataConsent ->
            "Llama model is not downloaded yet (waiting for download approval)."
        is ModelManager.State.Error -> "Llama is unavailable: ${state.message}"
        is ModelManager.State.Ready -> "Llama is ready."
    }

    /** ARGB_8888 software bitmap with EXIF orientation applied. */
    private fun decodeBitmap(uri: Uri): Bitmap {
        val resolver = getApplication<Application>().contentResolver

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        // Only very large photos are subsampled, to keep memory bounded.
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > MAX_SIDE) sample *= 2

        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inMutable = false
            inSampleSize = sample
        }
        val decoded = resolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        } ?: throw IllegalStateException("Could not read the selected image.")

        val orientation = try {
            resolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                )
            } ?: ExifInterface.ORIENTATION_NORMAL
        } catch (e: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.postRotate(90f); matrix.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.postRotate(270f); matrix.postScale(-1f, 1f) }
        }

        val rotated = if (matrix.isIdentity) decoded else {
            Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
                .also { if (it !== decoded) decoded.recycle() }
        }

        return if (rotated.config == Bitmap.Config.ARGB_8888) rotated else {
            rotated.copy(Bitmap.Config.ARGB_8888, false).also { rotated.recycle() }
        }
    }

    override fun onCleared() {
        adviceJob?.cancel()
        pipeline?.close()
        pipeline = null
        // ModelManager keeps the LlmModule for the whole process (load once).
        super.onCleared()
    }

    companion object {
        private const val TAG = "AANetraInspect"
        private const val MAX_SIDE = 4096

        fun percent(done: Long, total: Long): Int =
            if (total <= 0) 0 else ((done * 100) / total).toInt().coerceIn(0, 100)
    }
}
