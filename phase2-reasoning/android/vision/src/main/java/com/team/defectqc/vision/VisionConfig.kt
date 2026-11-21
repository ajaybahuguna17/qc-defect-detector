package com.team.defectqc.vision

/**
 * Every vision threshold of the app in one place.
 *
 * Fusion values mirror `FUSION_CONFIG` in phase1-vision/fusion.py (Phase 1.1 run
 * p11_split701515_det320_cls224_s42). `NOT_STEEL_THRESHOLD` 0.33 is the
 * val-chosen t_ns of the Phase 1.2 run p12_cls7_notsteel_s42 (results/phase1_2/fusion_config.json,
 * pinned by FusionConfigPinTest); it only matters with the 7-class classifier v2 (the 6-class
 * model never outputs not_steel).
 *
 * The unsure-HOLD values and [ImageCheckConfig] are app-only safety gates (not in fusion.py).
 */
object VisionConfig {

    /** Detector / classifier v1 class order (labels.txt). */
    val NEU_CLASSES = listOf(
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    )

    const val NOT_STEEL = "not_steel"

    /** Classifier v2 order: the 6 NEU classes, then not_steel. */
    val CLS7_CLASSES = NEU_CLASSES + NOT_STEEL

    // ---- fusion (fusion.py FUSION_CONFIG) ----
    const val CLASSIFIER_THRESHOLD = 0.64f

    /** Phase 1.2 run p12_cls7_notsteel_s42, fusion_config.json -> not_steel_threshold. */
    const val NOT_STEEL_THRESHOLD = 0.33f

    val DETECTOR_CLASS_THRESHOLDS = mapOf(
        "crazing" to 0.10f,
        "inclusion" to 0.50f,
        "patches" to 0.55f,
        "pitted_surface" to 0.15f,
        "rolled-in_scale" to 0.15f,
        "scratches" to 0.30f
    )

    const val NO_DEFECT_LABEL = "no defect found"
    const val NOT_STEEL_LABEL = "not a steel surface"
    const val IMAGE_CHECK_LABEL = "image not usable"
    const val REVIEW_LEVEL = "Review"

    // ---- unsure -> HOLD (app safety gate) ----
    /** Classifier did not fire and the best kept box is below this -> HOLD. */
    const val UNSURE_BEST_BOX_CONFIDENCE = 0.50f

    /** Kept boxes of this many (or more) different classes -> HOLD. */
    const val UNSURE_DISTINCT_CLASSES = 3

    // ---- input checks before vision ----
    val IMAGE_CHECK = ImageCheckConfig()
}

/** Thresholds of [ImageCheck]. Luma is 0..255, saturation 0..1. */
data class ImageCheckConfig(
    val minSidePx: Int = 64,
    val minMeanLuma: Double = 25.0,
    val maxMeanLuma: Double = 235.0,
    /**
     * Variance of the 3x3 Laplacian on the 256 px grayscale copy (gray 0..255). Deliberately
     * conservative: NEU steel textures are smooth, the 12 demo examples measure 9.6 (inclusion_117)
     * to 462 (crazing_11), and ImageCheckTest requires every one to stay >= 3x this value. It
     * catches flat / heavily defocused frames, not mild blur. Re-tune on Phase 3 camera images.
     */
    val minLaplacianVariance: Double = 3.0,
    val maxMeanSaturation: Double = 0.35,
    /** Longest side of the analysis copy. */
    val analysisSidePx: Int = 256
)
