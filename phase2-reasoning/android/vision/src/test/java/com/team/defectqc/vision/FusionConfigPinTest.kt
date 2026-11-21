package com.team.defectqc.vision

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/** VisionConfig must hold exactly the thresholds the Colab runs chose on val (fusion_config.json). */
class FusionConfigPinTest {

    private fun json(path: String): Map<*, *> =
        MiniJson.parse(File("../../../phase1-vision/results/$path").readText()) as Map<*, *>

    private fun f(v: Any?) = (v as Number).toFloat()

    @Test
    fun phase12Thresholds() {
        val c = json("phase1_2/fusion_config.json")
        assertEquals("val", c["selected_on"])
        assertEquals("p12_cls7_notsteel_s42", c["run_name"])
        assertEquals(f(c["not_steel_threshold"]), VisionConfig.NOT_STEEL_THRESHOLD, 0f)
        assertEquals(f(c["cls_threshold"]), VisionConfig.CLASSIFIER_THRESHOLD, 0f)
        assertEquals(VisionConfig.CLS7_CLASSES, c["classifier_classes"])
        val det = c["det_class_min_confidence"] as Map<*, *>
        assertEquals(VisionConfig.DETECTOR_CLASS_THRESHOLDS.keys, det.keys)
        for ((k, v) in VisionConfig.DETECTOR_CLASS_THRESHOLDS) assertEquals(k, f(det[k]), v, 0f)
    }

    @Test
    fun phase11DetectorThresholdsUnchanged() {
        val c = json("phase1_1/fusion_config.json")
        assertEquals(f(c["cls_threshold"]), VisionConfig.CLASSIFIER_THRESHOLD, 0f)
        val det = c["det_class_min_confidence"] as Map<*, *>
        for ((k, v) in VisionConfig.DETECTOR_CLASS_THRESHOLDS) assertEquals(k, f(det[k]), v, 0f)
    }
}
