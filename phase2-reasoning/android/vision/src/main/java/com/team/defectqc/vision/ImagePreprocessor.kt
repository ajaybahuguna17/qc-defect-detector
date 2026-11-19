
package com.team.defectqc.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.min
import kotlin.math.roundToInt

data class DetectorPreprocessedInput(
    val tensor: FloatArray,
    val scale: Float,
    val padX: Int,
    val padY: Int,
    val originalWidth: Int,
    val originalHeight: Int
)

object ImagePreprocessor {

    private const val DETECTOR_SIZE = 320
    private const val CLASSIFIER_SIZE = 224

    // Detector preprocessing:
    // Resize while preserving aspect ratio, then center-pad to 320x320.
    fun preprocessDetector(bitmap: Bitmap): DetectorPreprocessedInput {

        require(bitmap.width > 0 && bitmap.height > 0)

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        val scale = min(
            DETECTOR_SIZE.toFloat() / originalWidth,
            DETECTOR_SIZE.toFloat() / originalHeight
        )

        val resizedWidth =
            (originalWidth * scale).roundToInt().coerceAtLeast(1)

        val resizedHeight =
            (originalHeight * scale).roundToInt().coerceAtLeast(1)

        val padX = ((DETECTOR_SIZE - resizedWidth) / 2f - 0.1f)
            .roundToInt()

        val padY = ((DETECTOR_SIZE - resizedHeight) / 2f - 0.1f)
            .roundToInt()

        val resized = Bitmap.createScaledBitmap(
            bitmap,
            resizedWidth,
            resizedHeight,
            true
        )

        try {
            val letterboxed = Bitmap.createBitmap(
                DETECTOR_SIZE,
                DETECTOR_SIZE,
                Bitmap.Config.ARGB_8888
            )

            try {
                val canvas = Canvas(letterboxed)

                canvas.drawColor(Color.rgb(114, 114, 114))

                val paint = Paint(Paint.FILTER_BITMAP_FLAG)

                canvas.drawBitmap(
                    resized,
                    padX.toFloat(),
                    padY.toFloat(),
                    paint
                )

                return DetectorPreprocessedInput(
                    tensor = bitmapToNchw(letterboxed),
                    scale = scale,
                    padX = padX,
                    padY = padY,
                    originalWidth = originalWidth,
                    originalHeight = originalHeight
                )

            } finally {
                letterboxed.recycle()
            }

        } finally {
            if (resized !== bitmap) {
                resized.recycle()
            }
        }
    }

    // Classifier preprocessing:
    // Resize shorter side to 224, then take a center crop of 224x224.
    fun preprocessClassifier(bitmap: Bitmap): FloatArray {

        require(bitmap.width > 0 && bitmap.height > 0)

        val width = bitmap.width
        val height = bitmap.height

        val resizedWidth: Int
        val resizedHeight: Int

        if (width < height) {
            resizedWidth = CLASSIFIER_SIZE
            resizedHeight =
                (height * CLASSIFIER_SIZE.toFloat() / width).toInt()
        } else {
            resizedHeight = CLASSIFIER_SIZE
            resizedWidth =
                (width * CLASSIFIER_SIZE.toFloat() / height).toInt()
        }

        val resized = Bitmap.createScaledBitmap(
            bitmap,
            resizedWidth,
            resizedHeight,
            true
        )

        try {
            val cropX =
                ((resizedWidth - CLASSIFIER_SIZE) / 2.0).roundToInt()

            val cropY =
                ((resizedHeight - CLASSIFIER_SIZE) / 2.0).roundToInt()

            val cropped = Bitmap.createBitmap(
                resized,
                cropX,
                cropY,
                CLASSIFIER_SIZE,
                CLASSIFIER_SIZE
            )

            try {
                return bitmapToNchw(cropped)
            } finally {
                if (cropped !== resized) {
                    cropped.recycle()
                }
            }

        } finally {
            if (resized !== bitmap) {
                resized.recycle()
            }
        }
    }

    // Converts RGB pixels to normalized NCHW float format.
    private fun bitmapToNchw(bitmap: Bitmap): FloatArray {

        val width = bitmap.width
        val height = bitmap.height
        val pixelCount = width * height

        val pixels = IntArray(pixelCount)

        bitmap.getPixels(
            pixels,
            0,
            width,
            0,
            0,
            width,
            height
        )

        val output = FloatArray(3 * pixelCount)

        for (i in pixels.indices) {

            val pixel = pixels[i]

            val red = Color.red(pixel) / 255.0f
            val green = Color.green(pixel) / 255.0f
            val blue = Color.blue(pixel) / 255.0f

            output[i] = red
            output[pixelCount + i] = green
            output[2 * pixelCount + i] = blue
        }

        return output
    }
}
