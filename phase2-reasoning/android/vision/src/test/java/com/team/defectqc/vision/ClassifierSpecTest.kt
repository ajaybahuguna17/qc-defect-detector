package com.team.defectqc.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

class ClassifierSpecTest {

    private fun modelIo(classes: List<String>, file: String = "defect_cls_yolov8n_224_fp32.tflite", n: Int = classes.size) = """
        {"task": "classify", "imgsz": 224,
         "classes": [${classes.joinToString(",") { "\"$it\"" }}],
         "preprocessing": {"resize": "shorter side -> 224", "mean": [0.0, 0.0, 0.0]},
         "models": {
           "fp32": {"input_shape": [1, 3, 224, 224], "input_quant": [0.0, 0], "output_shape": [1, $n],
                    "status": "ok", "file": "$file", "size_bytes": 5815905,
                    "probe": {"image": "crazing_1.jpg", "probs": [0.1, 2e-3, -1.5E-4]}},
           "int8": {"status": "skipped", "reason": "hung \"45 min\"\nin calibration", "timestamp": "2026-11-07"}
         }}
    """.trimIndent()

    private fun labels(classes: List<String>) = classes.joinToString("\n") + "\n"

    @Test
    fun sixClassV1() {
        val s = ClassifierSpec.parse(modelIo(VisionConfig.NEU_CLASSES), labels(VisionConfig.NEU_CLASSES))
        assertEquals(6, s.numClasses)
        assertFalse(s.hasNotSteel)
        assertEquals("classifier/defect_cls_yolov8n_224_fp32.tflite", s.assetPath)
    }

    @Test
    fun sevenClassV2() {
        val s = ClassifierSpec.parse(modelIo(VisionConfig.CLS7_CLASSES, "cls7.tflite"), labels(VisionConfig.CLS7_CLASSES))
        assertEquals(7, s.numClasses)
        assertTrue(s.hasNotSteel)
        assertEquals(VisionConfig.NOT_STEEL, s.classes.last())
        assertEquals("classifier/cls7.tflite", s.assetPath)
    }

    @Test(expected = IllegalArgumentException::class)
    fun wrongOrderRejected() {
        val alpha = VisionConfig.CLS7_CLASSES.sorted()   // not_steel at index 2 (ImageFolder order)
        ClassifierSpec.parse(modelIo(alpha), labels(alpha))
    }

    @Test(expected = IllegalArgumentException::class)
    fun labelsAndModelIoMustAgree() {
        ClassifierSpec.parse(modelIo(VisionConfig.NEU_CLASSES), labels(VisionConfig.CLS7_CLASSES))
    }

    @Test(expected = IllegalArgumentException::class)
    fun outputShapeMustMatchLabels() {
        ClassifierSpec.parse(modelIo(VisionConfig.CLS7_CLASSES, n = 6), labels(VisionConfig.CLS7_CLASSES))
    }

    @Test(expected = IllegalArgumentException::class)
    fun pathInFileNameRejected() {
        ClassifierSpec.parse(modelIo(VisionConfig.NEU_CLASSES, "../x.tflite"), labels(VisionConfig.NEU_CLASSES))
    }

    @Test
    fun miniJsonBasics() {
        val v = MiniJson.parse("""{"a": [1, -2.5e1, true, false, null], "b": {"c": "xA\n"}}""") as Map<*, *>
        assertEquals(listOf(1.0, -25.0, true, false, null), v["a"])
        assertEquals("xA\n", (v["b"] as Map<*, *>)["c"])
    }

    /** The real exported files (gitignored; skipped when not present locally). */
    @Test
    fun realExportedClassifiers() {
        val root = File("../../../phase1-vision/models/exported")
        var checked = 0
        for ((dir, n) in listOf("phase1_1" to 6, "phase1_2" to 7)) {
            val d = root.resolve("$dir/classifier")
            if (!d.resolve("model_io.json").isFile) continue
            val s = ClassifierSpec.parse(d.resolve("model_io.json").readText(), d.resolve("labels.txt").readText())
            assertEquals(n, s.numClasses)
            assertTrue(d.resolve(s.fileName).isFile)
            checked++
        }
        assumeTrue("no exported classifier found locally", checked > 0)
    }
}
