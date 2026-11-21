package com.team.defectqc.advisor

import android.os.SystemClock
import android.util.Log
import com.team.defectqc.core.Advisor
import com.team.defectqc.core.Advisory
import com.team.defectqc.core.DetectionEvent
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.util.concurrent.Executor

class LlamaAdvisor(
    private val modelProvider: () -> LlmModule?,
    generationExecutor: Executor,
    // Total length (prompt + answer); must match the .pte max context (512).
    private val sequenceLength: Int = 512
) : Advisor {

    private val dispatcher: CoroutineDispatcher =
        generationExecutor.asCoroutineDispatcher()

    private val cache = AdvisoryCache()

    override suspend fun load(): Boolean = withContext(dispatcher) {
        modelProvider() != null
    }


override suspend fun advise(event: DetectionEvent): Advisory =
    withContext(dispatcher) {

        require(event.alert) {
            "Clean inspections must bypass the advisor."
        }

        val t0 = SystemClock.elapsedRealtime()

        // Step 1: cache key = final class + severity + decision + defects-found bucket
        val key = cache.key(event)

        // Step 2: cached advisory (no Llama call)
        cache.get(key)?.let {
            val hit = it.copy(
                eventId = event.id,
                cached = true,
                llmMs = SystemClock.elapsedRealtime() - t0,
                tokensPerSecond = 0f,
                promptTokens = null,
                generatedTokens = null,
                ttftMs = null,
                stoppedEarly = false
            )
            logAdvice(hit)
            return@withContext hit
        }

        // Step 3: Get Llama model only when cache misses
        val model = checkNotNull(modelProvider()) {
            "Load the Llama model before requesting an advisory."
        }

        // Step 4: Build prompt
        val prompt = PromptBuilder.inspection(event)

        // Step 5: Run Llama, stopping right after the ACTION line
        val result = generate(model, prompt, stopAfterAction = true)
        val stats = GenerationStats.parse(result.stats)

        // Step 6: Parse response (parser unchanged)
        val advisory = AdvisoryParser.parse(
            event = event,
            raw = result.text,
            llmMs = result.elapsedMs,
            tokensPerSecond = stats.tokensPerSecond
        ).copy(
            promptTokens = stats.promptTokens,
            generatedTokens = stats.generatedTokens,
            ttftMs = stats.ttftMs ?: result.firstChunkMs,
            stoppedEarly = result.stoppedEarly
        )

        // Step 7: Save advisory
        cache.put(key, advisory)
        logAdvice(advisory)

        advisory
    }

    private fun logAdvice(a: Advisory) {
        Log.i(TAG, adviceLogLine(a))
    }


    override suspend fun ask(
        question: String,
        logSummary: String
    ): String = withContext(dispatcher) {

        require(question.isNotBlank()) {
            "Question must not be blank."
        }

        val model = checkNotNull(modelProvider()) {
            "Load the Llama model first."
        }

        generate(
            model,
            PromptBuilder.question(question, logSummary),
            stopAfterAction = false
        ).text.trim().ifBlank {
            "I could not generate an answer. Please try again."
        }
    }

    override suspend fun close() {
        // ChatViewModel owns the shared model and executor.
        // Do not close either from this advisor.
        cache.clear()
    }

    private data class GenerationResult(
        val text: String,
        val stats: String,
        val elapsedMs: Long,
        /** Wall-clock ms to the first streamed chunk (TTFT fallback when stats lack it). */
        val firstChunkMs: Long?,
        val stoppedEarly: Boolean
    )

    /**
     * Blocking generation. With [stopAfterAction], LlmModule.stop() (ExecuTorch 1.5.1) is called
     * from the callback as soon as the ACTION line is complete ([AdviceStopRule]); chunks that
     * still arrive after that are ignored. [sequenceLength] (512) stays the upper bound, and if
     * the model never writes ACTION it runs to end-of-turn / the limit as before.
     */
    private fun generate(
        model: LlmModule,
        prompt: String,
        stopAfterAction: Boolean
    ): GenerationResult {

        val output = StringBuilder()
        var stats = ""
        var error: String? = null
        val stopRule = AdviceStopRule()
        var firstChunkMs: Long? = null

        model.resetContext()

        val start = SystemClock.elapsedRealtime()

        model.generate(
            prompt,
            sequenceLength,
            object : LlmCallback {

                override fun onResult(result: String) {
                    if (firstChunkMs == null) firstChunkMs = SystemClock.elapsedRealtime() - start
                    if (stopRule.stopped) return   // tokens still in flight after stop()
                    output.append(result)
                    if (stopAfterAction && stopRule.onChunk(result)) {
                        model.stop()
                    }
                }

                override fun onStats(result: String) {
                    stats = result
                }

                override fun onError(code: Int, message: String) {
                    error = "$code: $message"
                }
            },
            false
        )

        val elapsed = SystemClock.elapsedRealtime() - start
        Log.i(TAG, "Generation stats: $stats")
Log.i(TAG, "Generation elapsed: $elapsed ms")

        error?.let {
            throw IllegalStateException(it)
        }
        if (output.isBlank()) {
    throw IllegalStateException("Llama returned an empty response.")
}

        return GenerationResult(
            text = output.toString(),
            stats = stats,
            elapsedMs = elapsed,
            firstChunkMs = firstChunkMs,
            stoppedEarly = stopRule.stopped
        )
    }

    companion object {
        private const val TAG = "LlamaAdvisor"

        /** One log line per advice: prompt / generated tokens, TTFT, total ms, cached. */
        fun adviceLogLine(a: Advisory): String =
            "advice ${a.finalClass}/${a.severityLevel}: prompt_tokens=${a.promptTokens ?: "n/a"} " +
                "generated_tokens=${a.generatedTokens ?: "n/a"} ttft_ms=${a.ttftMs ?: "n/a"} " +
                "total_ms=${a.llmMs} cached=${if (a.cached) "yes" else "no"} " +
                "stopped_after_action=${if (a.stoppedEarly) "yes" else "no"}"
    }
}