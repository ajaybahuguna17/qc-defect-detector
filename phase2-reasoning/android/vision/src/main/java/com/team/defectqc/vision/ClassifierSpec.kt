package com.team.defectqc.vision

/**
 * Which classifier is in the assets: file name, class list and class count, read from
 * `classifier/model_io.json` + `classifier/labels.txt` (written by export_cls_tflite.py).
 *
 * Supported: v1 = the 6 NEU classes (Phase 1.1), v2 = the 6 NEU classes then not_steel (Phase 1.2).
 * Anything else (other order, other classes, labels.txt and model_io.json disagreeing, a missing
 * FP32 entry, an output shape other than [1, n]) throws, so a wrong model never loads silently.
 */
data class ClassifierSpec(
    val fileName: String,
    val classes: List<String>,
    val inputShape: List<Int>
) {
    val numClasses: Int get() = classes.size
    val hasNotSteel: Boolean get() = VisionConfig.NOT_STEEL in classes
    val assetPath: String get() = "$ASSET_DIR/$fileName"

    companion object {
        const val ASSET_DIR = "classifier"
        const val MODEL_IO = "$ASSET_DIR/model_io.json"
        const val LABELS = "$ASSET_DIR/labels.txt"

        fun parse(modelIoJson: String, labelsTxt: String): ClassifierSpec {
            val labels = labelsTxt.lines().map { it.trim() }.filter { it.isNotEmpty() }
            require(labels == VisionConfig.NEU_CLASSES || labels == VisionConfig.CLS7_CLASSES) {
                "labels.txt $labels is neither the 6 NEU classes nor the 6 NEU classes + not_steel"
            }

            val root = MiniJson.parse(modelIoJson) as? Map<*, *>
                ?: throw IllegalArgumentException("model_io.json is not a JSON object")
            (root["classes"] as? List<*>)?.let { cls ->
                require(cls == labels) { "model_io.json classes $cls != labels.txt $labels" }
            }
            (root["task"] as? String)?.let { require(it == "classify") { "model_io.json task is '$it'" } }

            val fp32 = ((root["models"] as? Map<*, *>)?.get("fp32") as? Map<*, *>)
                ?: throw IllegalArgumentException("model_io.json has no models.fp32 entry")
            val status = fp32["status"] as? String ?: "ok"
            require(status == "ok") { "classifier fp32 status is '$status'" }
            val file = fp32["file"] as? String
                ?: throw IllegalArgumentException("model_io.json models.fp32 has no file")
            require(file.isNotBlank() && '/' !in file && '\\' !in file && !file.startsWith(".")) {
                "unexpected classifier file name '$file'"
            }
            (fp32["output_shape"] as? List<*>)?.let { shape ->
                require(shape.map { (it as Number).toInt() } == listOf(1, labels.size)) {
                    "model_io.json output_shape $shape != [1, ${labels.size}]"
                }
            }
            val input = (fp32["input_shape"] as? List<*>)?.map { (it as Number).toInt() } ?: listOf(1, 3, 224, 224)
            require(input == listOf(1, 3, 224, 224)) { "classifier input shape $input != [1, 3, 224, 224]" }
            return ClassifierSpec(file, labels, input)
        }
    }
}

/** Minimal JSON reader (objects, arrays, strings, numbers, true/false/null). No Android deps. */
internal object MiniJson {

    fun parse(text: String): Any? {
        val p = Parser(text)
        val v = p.value()
        p.ws()
        require(p.i == text.length) { "trailing characters at ${p.i}" }
        return v
    }

    private class Parser(val s: String) {
        var i = 0

        fun ws() {
            while (i < s.length && s[i].isWhitespace()) i++
        }

        fun value(): Any? {
            ws()
            require(i < s.length) { "unexpected end of JSON" }
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> lit("true", true)
                'f' -> lit("false", false)
                'n' -> lit("null", null)
                else -> if (c == '-' || c.isDigit()) num() else throw IllegalArgumentException("bad JSON at $i: '$c'")
            }
        }

        private fun lit(word: String, v: Any?): Any? {
            require(s.startsWith(word, i)) { "bad JSON literal at $i" }
            i += word.length
            return v
        }

        private fun num(): Double {
            val start = i
            while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
            return s.substring(start, i).toDouble()
        }

        private fun str(): String {
            i++ // opening quote
            val sb = StringBuilder()
            while (true) {
                require(i < s.length) { "unterminated JSON string" }
                val c = s[i++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        val e = s[i++]
                        when (e) {
                            '"', '\\', '/' -> sb.append(e)
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> { sb.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                            else -> throw IllegalArgumentException("bad escape \\$e")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun arr(): List<Any?> {
            i++
            val out = mutableListOf<Any?>()
            ws()
            if (s[i] == ']') { i++; return out }
            while (true) {
                out.add(value())
                ws()
                when (s[i++]) {
                    ',' -> continue
                    ']' -> return out
                    else -> throw IllegalArgumentException("expected , or ] at ${i - 1}")
                }
            }
        }

        private fun obj(): Map<String, Any?> {
            i++
            val out = LinkedHashMap<String, Any?>()
            ws()
            if (s[i] == '}') { i++; return out }
            while (true) {
                ws()
                require(s[i] == '"') { "expected key at $i" }
                val k = str()
                ws()
                require(s[i++] == ':') { "expected : at ${i - 1}" }
                out[k] = value()
                ws()
                when (s[i++]) {
                    ',' -> continue
                    '}' -> return out
                    else -> throw IllegalArgumentException("expected , or } at ${i - 1}")
                }
            }
        }
    }
}
