package com.team.defectqc.advisor

/**
 * Early stop for inspection advice: the answer format is SUMMARY / CAUSE / ACTION, so once the
 * ACTION line is complete there is nothing more to read. Generation is stopped (LlmModule.stop())
 * as soon as the streamed text holds an ACTION line with text that is ended by a newline or by
 * the Llama 3 end-of-turn token. If the model never writes ACTION, nothing stops early and the
 * old behaviour applies (end-of-turn or the sequence-length limit).
 *
 * The ACTION pattern is the parser's (AdvisoryParser): a line starting with "ACTION:" (any case,
 * leading spaces allowed). Pure Kotlin, unit-tested.
 */
class AdviceStopRule {

    private val buffer = StringBuilder()

    /** Text seen so far (everything appended, including the chunk that completed ACTION). */
    val text: String get() = buffer.toString()

    var stopped: Boolean = false
        private set

    /** Append one streamed chunk; true when generation should stop now (returned once). */
    fun onChunk(chunk: String): Boolean {
        if (stopped) return false
        buffer.append(chunk)
        if (isComplete(buffer)) {
            stopped = true
            return true
        }
        return false
    }

    companion object {
        const val EOT = "<|eot_id|>"

        private val ACTION_LINE = Regex("^\\s*ACTION\\s*:\\s*(.*)$", RegexOption.IGNORE_CASE)

        /** True when [text] contains a complete, non-empty ACTION line. */
        fun isComplete(text: CharSequence): Boolean {
            val s = text.toString()
            val eot = s.indexOf(EOT)
            val body = if (eot >= 0) s.substring(0, eot) else s
            val lines = body.split('\n')
            // Every line except the last is terminated by '\n'; the last one only by EOT (if present).
            lines.forEachIndexed { i, line ->
                val terminated = i < lines.size - 1 || eot >= 0
                if (terminated) {
                    val m = ACTION_LINE.find(line.trimEnd('\r'))
                    if (m != null && m.groupValues[1].isNotBlank()) return true
                }
            }
            return false
        }
    }
}

/** Numbers from the ExecuTorch LlmCallback.onStats JSON (flat object of numbers). */
data class GenerationStats(
    val promptTokens: Int?,
    val generatedTokens: Int?,
    /** first_token_ms - inference_start_ms, when both are present. */
    val ttftMs: Long?,
    /** generated tokens per second of the decode phase (after prompt eval). */
    val tokensPerSecond: Float
) {
    companion object {
        private fun num(json: String, key: String): Double? =
            Regex("\"$key\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)").find(json)
                ?.groupValues?.get(1)?.toDoubleOrNull()

        fun parse(json: String): GenerationStats {
            val prompt = num(json, "prompt_tokens")
            val gen = num(json, "generated_tokens")
            val start = num(json, "inference_start_ms")
            val first = num(json, "first_token_ms")
            val promptEnd = num(json, "prompt_eval_end_ms")
            val end = num(json, "inference_end_ms")
            val tps = if (gen != null && gen > 0 && end != null && promptEnd != null && end > promptEnd) {
                (gen * 1000.0 / (end - promptEnd)).toFloat()
            } else 0f
            val ttft = if (start != null && first != null && first >= start) (first - start).toLong() else null
            return GenerationStats(prompt?.toInt(), gen?.toInt(), ttft, tps)
        }
    }
}
