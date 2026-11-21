package com.team.defectqc.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Locale
import kotlin.random.Random

class ImageCheckTest {

    /** Module dir is the test working directory: android/vision -> repo root is ../../.. */
    private val examplesDir = File("../../../phase1-vision/demo/examples")

    /**
     * JPEG -> ARGB pixels with the desktop JDK's javax.imageio. It is not on the Android
     * unit-test compile classpath (android.jar), but it is in the JVM that runs the tests,
     * so it is called by reflection.
     */
    private fun load(f: File): Triple<Int, Int, IntArray> {
        val imageIO = Class.forName("javax.imageio.ImageIO")
        val img = imageIO.getMethod("read", File::class.java).invoke(null, f) ?: error("cannot read $f")
        val c = img.javaClass
        val w = c.getMethod("getWidth").invoke(img) as Int
        val h = c.getMethod("getHeight").invoke(img) as Int
        val getRgb = c.getMethod(
            "getRGB", Int::class.javaPrimitiveType, Int::class.javaPrimitiveType, Int::class.javaPrimitiveType,
            Int::class.javaPrimitiveType, IntArray::class.java, Int::class.javaPrimitiveType, Int::class.javaPrimitiveType
        )
        val px = getRgb.invoke(img, 0, 0, w, h, null, 0, w) as IntArray
        return Triple(w, h, px)
    }

    private fun check(w: Int, h: Int, px: IntArray) = ImageCheck.check(w, h, px)

    private fun solid(w: Int, h: Int, r: Int, g: Int, b: Int) =
        IntArray(w * h) { (0xFF shl 24) or (r shl 16) or (g shl 8) or b }

    /** Gray noise texture (sharp), mean ~ [mean]. */
    private fun noise(w: Int, h: Int, mean: Int, amp: Int, seed: Int = 1): IntArray {
        val rnd = Random(seed)
        return IntArray(w * h) {
            val v = (mean + rnd.nextInt(-amp, amp + 1)).coerceIn(0, 255)
            (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
    }

    @Test
    fun allTwelveDemoExamplesPass() {
        val files = examplesDir.listFiles { f -> f.extension.lowercase() == "jpg" }?.sortedBy { it.name }
            ?: error("examples not found at ${examplesDir.absolutePath}")
        assertEquals(12, files.size)
        for (f in files) {
            val (w, h, px) = load(f)
            val r = check(w, h, px)
            val m = r.metrics!!
            println(
                String.format(
                    Locale.US, "%-26s luma %6.1f  sat %.3f  lapVar %9.1f  -> %s",
                    f.name, m.meanLuma, m.meanSaturation, m.laplacianVariance, if (r.passed) "PASS" else r.failure
                )
            )
            assertTrue("${f.name}: ${r.reason}", r.passed)
            // margins: the blur threshold must stay well below every example
            assertTrue("${f.name} sharpness ${m.laplacianVariance} too close to the threshold",
                m.laplacianVariance >= 3 * VisionConfig.IMAGE_CHECK.minLaplacianVariance)
        }
    }

    @Test
    fun tooSmall() {
        val r = check(63, 200, noise(63, 200, 120, 60))
        assertFalse(r.passed)
        assertEquals("too_small", r.failure)
        assertTrue(r.reason.contains("too small"))
        assertTrue(check(64, 64, noise(64, 64, 120, 60)).passed)
    }

    @Test
    fun tooDarkAndTooBright() {
        assertEquals("too_dark", check(200, 200, noise(200, 200, 12, 10)).failure)
        assertEquals("too_bright", check(200, 200, noise(200, 200, 245, 8)).failure)
    }

    @Test
    fun tooColourful() {
        val rnd = Random(3)
        val px = IntArray(200 * 200) {
            val n = rnd.nextInt(-40, 41)
            (0xFF shl 24) or ((200 + n / 4).coerceIn(0, 255) shl 16) or ((60 + n).coerceIn(0, 255) shl 8) or 40
        }
        val r = check(200, 200, px)
        assertEquals("too_colourful", r.failure)
        assertTrue(r.metrics!!.meanSaturation > 0.35)
    }

    @Test
    fun tooBlurry() {
        // smooth horizontal gradient: no edges at all
        val px = IntArray(300 * 200) { i ->
            val v = 60 + (i % 300) * 120 / 300
            (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        val r = check(300, 200, px)
        assertEquals("too_blurry", r.failure)
        assertTrue(r.reason.contains("blurry"))
    }

    @Test
    fun uniformGreyIsBlurry() {
        assertEquals("too_blurry", check(400, 400, solid(400, 400, 128, 128, 128)).failure)
    }

    @Test
    fun sharpGreyTexturePasses() {
        assertTrue(check(640, 480, noise(640, 480, 120, 50)).passed)
    }

    @Test
    fun resizeLongestSideIs256() {
        val (w, h, px) = ImageCheck.resizeLongestSide(noise(1024, 512, 100, 20), 1024, 512, 256)
        assertEquals(256, w); assertEquals(128, h); assertEquals(256 * 128, px.size)
        val (w2, h2, _) = ImageCheck.resizeLongestSide(noise(200, 100, 100, 20), 200, 100, 256)
        assertEquals(256, w2); assertEquals(128, h2)
    }

    @Test
    fun laplacianOfConstantIsZero() {
        assertEquals(0.0, ImageCheck.laplacianVariance(DoubleArray(25) { 7.0 }, 5, 5), 1e-12)
    }
}
