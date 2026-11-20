package com.team.defectqc.advisor

import android.os.SystemClock
import android.util.Log
import com.team.defectqc.core.Advisor
import com.team.defectqc.core.Advisory
import com.team.defectqc.core.DetectionEvent
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.util.concurrent.Executor

class LlamaAdvisor(
    private val modelProvider: () -> LlmModule?,
    generationExecutor: Executor,
    private val sequenceLength: Int = 80
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

        // Step 1: Generate cache key
        val key = cache.key(event)

        // Step 2: Return cached advisory if available
        cache.get(key)?.let {
            return@withContext it.copy(
                eventId = event.id,
                cached = true,
                llmMs = 0L
            )
        }

        // Step 3: Get Llama model only when cache misses
        val model = checkNotNull(modelProvider()) {
            "Load the Llama model before requesting an advisory."
        }

        // Step 4: Build prompt
        val prompt = PromptBuilder.inspection(event)

        // Step 5: Run Llama inference
        val result = generate(model, prompt)

        // Step 6: Parse response
        val advisory = AdvisoryParser.parse(
            event = event,
            raw = result.text,
            llmMs = result.elapsedMs,
            tokensPerSecond = tokensPerSecond(result.stats)
        )

        // Step 7: Save advisory
        cache.put(key, advisory)

        advisory
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
            PromptBuilder.question(question, logSummary)
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
        val elapsedMs: Long
    )

    private fun generate(
        model: LlmModule,
        prompt: String
    ): GenerationResult {

        val output = StringBuilder()
        var stats = ""
        var error: String? = null

        model.resetContext()

        val start = SystemClock.elapsedRealtime()

        model.generate(
            prompt,
            sequenceLength,
            object : LlmCallback {

                override fun onResult(result: String) {
                    output.append(result)
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

        error?.let {
            throw IllegalStateException(it)
        }
        if (output.isBlank()) {
    throw IllegalStateException("Llama returned an empty response.")
}

        return GenerationResult(
            text = output.toString(),
            stats = stats,
            elapsedMs = elapsed
        )
    }

    private fun tokensPerSecond(json: String): Float = try {
        val data = JSONObject(json)

        val tokens = data.optDouble("generated_tokens", 0.0)
        val end = data.optDouble("inference_end_ms", 0.0)
        val promptEnd = data.optDouble("prompt_eval_end_ms", 0.0)

        val duration = end - promptEnd

        if (tokens > 0 && duration > 0) {
            (tokens * 1000.0 / duration).toFloat()
        } else {
            0f
        }

    } catch (e: Exception) {
        Log.w(TAG, "Could not parse generation statistics", e)
        0f
    }

    companion object {
        private const val TAG = "LlamaAdvisor"
    }
}