package com.team.defectqc.vision

import java.util.Locale
import kotlin.math.max
import kotlin.math.min

/**
 * Input checks that run BEFORE the vision models (app safety gate, not in fusion.py).
 * A failed check -> decision RETAKE with [ImageCheckResult.reason]; the models are not run.
 *
 * Checks, in order: too small (shorter side), too dark / too bright (mean luma), too colourful
 * (mean HSV saturation), too blurry (variance of the 3x3 Laplacian). Brightness, saturation and
 * blur are measured on a copy whose longest side is [ImageCheckConfig.analysisSidePx] (256 px),
 * so the numbers do not depend on the camera resolution.
 *
 * Pure Kotlin on ARGB pixels (Bitmap.getPixels layout), so it is unit-tested on the JVM.
 */
object ImageCheck {

    data class Metrics(
        val width: Int,
        val height: Int,
        val meanLuma: Double,
        val meanSaturation: Double,
        val laplacianVariance: Double
    )

    data class ImageCheckResult(
        val passed: Boolean,
        /** Machine-readable failed check: "too_small", "too_dark", ... ; null when passed. */
        val failure: String?,
        val reason: String,
        val metrics: Metrics?
    )

    /**
     * @param width original image width in pixels
     * @param height original image height in pixels
     * @param argb pixels of the image or of an already downscaled copy ([argbWidth] x [argbHeight])
     */
    fun check(
        width: Int,
        height: Int,
        argb: IntArray,
        argbWidth: Int = width,
        argbHeight: Int = height,
        config: ImageCheckConfig = VisionConfig.IMAGE_CHECK
    ): ImageCheckResult {
        require(argb.size == argbWidth * argbHeight) { "pixel array does not match $argbWidth x $argbHeight" }

        if (min(width, height) < config.minSidePx) {
            return fail(
                "too_small",
                "Image too small (${width}x$height px; each side must be at least ${config.minSidePx} px).",
                null
            )
        }

        val (w, h, small) = resizeLongestSide(argb, argbWidth, argbHeight, config.analysisSidePx)
        val m = measure(width, height, small, w, h)

        return when {
            m.meanLuma < config.minMeanLuma -> fail(
                "too_dark",
                "Image too dark (mean brightness ${fmt(m.meanLuma, 0)} < ${fmt(config.minMeanLuma, 0)} of 255).", m
            )
            m.meanLuma > config.maxMeanLuma -> fail(
                "too_bright",
                "Image too bright (mean brightness ${fmt(m.meanLuma, 0)} > ${fmt(config.maxMeanLuma, 0)} of 255).", m
            )
            m.meanSaturation > config.maxMeanSaturation -> fail(
                "too_colourful",
                "Image too colourful for a bare steel surface (mean saturation ${fmt(m.meanSaturation, 2)} > " +
                    "${fmt(config.maxMeanSaturation, 2)}).", m
            )
            m.laplacianVariance < config.minLaplacianVariance -> fail(
                "too_blurry",
                "Image too blurry (sharpness ${fmt(m.laplacianVariance, 1)} < ${fmt(config.minLaplacianVariance, 1)}). " +
                    "Hold the camera still and focus on the surface.", m
            )
            else -> ImageCheckResult(true, null, "Image checks passed.", m)
        }
    }

    fun measure(width: Int, height: Int, argb: IntArray, w: Int, h: Int): Metrics {
        val gray = DoubleArray(w * h)
        var lumaSum = 0.0
        var satSum = 0.0
        for (i in argb.indices) {
            val p = argb[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            val y = 0.299 * r + 0.587 * g + 0.114 * b
            gray[i] = y
            lumaSum += y
            val mx = max(r, max(g, b))
            val mn = min(r, min(g, b))
            satSum += if (mx == 0) 0.0 else (mx - mn).toDouble() / mx
        }
        val n = argb.size.toDouble()
        return Metrics(width, height, lumaSum / n, satSum / n, laplacianVariance(gray, w, h))
    }

    /** Variance of the 3x3 Laplacian [0 1 0; 1 -4 1; 0 1 0] over the interior pixels. */
    fun laplacianVariance(gray: DoubleArray, w: Int, h: Int): Double {
        if (w < 3 || h < 3) return 0.0
        var sum = 0.0
        var sumSq = 0.0
        var count = 0
        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val i = row + x
                val lap = gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w] - 4.0 * gray[i]
                sum += lap
                sumSq += lap * lap
                count++
            }
        }
        val mean = sum / count
        return sumSq / count - mean * mean
    }

    /**
     * ARGB copy whose longest side is [side]: box average when shrinking, bilinear when enlarging.
     * Returns (width, height, pixels).
     */
    fun resizeLongestSide(argb: IntArray, w: Int, h: Int, side: Int): Triple<Int, Int, IntArray> {
        val scale = side.toDouble() / max(w, h)
        val nw = max(1, Math.round(w * scale).toInt())
        val nh = max(1, Math.round(h * scale).toInt())
        if (nw == w && nh == h) return Triple(w, h, argb)
        val out = IntArray(nw * nh)
        val sx = w.toDouble() / nw
        val sy = h.toDouble() / nh
        if (sx > 1.0 || sy > 1.0) {
            for (oy in 0 until nh) {
                val y0 = (oy * sy).toInt()
                val y1 = max(y0 + 1, min(h, ((oy + 1) * sy).toInt()))
                for (ox in 0 until nw) {
                    val x0 = (ox * sx).toInt()
                    val x1 = max(x0 + 1, min(w, ((ox + 1) * sx).toInt()))
                    var r = 0L; var g = 0L; var b = 0L
                    for (y in y0 until y1) for (x in x0 until x1) {
                        val p = argb[y * w + x]
                        r += (p shr 16) and 0xFF; g += (p shr 8) and 0xFF; b += p and 0xFF
                    }
                    val c = ((y1 - y0) * (x1 - x0)).toLong()
                    out[oy * nw + ox] = pack((r / c).toInt(), (g / c).toInt(), (b / c).toInt())
                }
            }
        } else {
            for (oy in 0 until nh) {
                val fy = ((oy + 0.5) * sy - 0.5).coerceIn(0.0, (h - 1).toDouble())
                val y0 = fy.toInt(); val y1 = min(h - 1, y0 + 1); val dy = fy - y0
                for (ox in 0 until nw) {
                    val fx = ((ox + 0.5) * sx - 0.5).coerceIn(0.0, (w - 1).toDouble())
                    val x0 = fx.toInt(); val x1 = min(w - 1, x0 + 1); val dx = fx - x0
                    val p00 = argb[y0 * w + x0]; val p01 = argb[y0 * w + x1]
                    val p10 = argb[y1 * w + x0]; val p11 = argb[y1 * w + x1]
                    fun ch(shift: Int): Int {
                        val a = ((p00 shr shift) and 0xFF) * (1 - dx) + ((p01 shr shift) and 0xFF) * dx
                        val b = ((p10 shr shift) and 0xFF) * (1 - dx) + ((p11 shr shift) and 0xFF) * dx
                        return Math.round(a * (1 - dy) + b * dy).toInt()
                    }
                    out[oy * nw + ox] = pack(ch(16), ch(8), ch(0))
                }
            }
        }
        return Triple(nw, nh, out)
    }

    private fun pack(r: Int, g: Int, b: Int): Int = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    private fun fail(code: String, reason: String, m: Metrics?) = ImageCheckResult(false, code, reason, m)

    private fun fmt(v: Double, decimals: Int) = String.format(Locale.US, "%.${decimals}f", v)
}
