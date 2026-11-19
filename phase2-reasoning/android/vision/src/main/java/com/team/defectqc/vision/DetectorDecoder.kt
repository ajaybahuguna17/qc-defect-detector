
package com.team.defectqc.vision

import kotlin.math.max
import kotlin.math.min

object DetectorDecoder {

    private const val IMAGE_SIZE = 320
    private const val NUM_ANCHORS = 2100
    private const val NUM_CLASSES = 6

    private const val DECODE_CONFIDENCE = 0.05f
    private const val NMS_IOU = 0.45f
    private const val MAX_DETECTIONS = 300

    private val classNames = listOf(
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    )

    private data class Candidate(
        val anchorIndex: Int,
        val classIndex: Int,
        val confidence: Float,
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float
    )

    fun decode(
        output: FloatArray,
        transform: DetectorPreprocessedInput
    ): List<VisionBox> {

        require(output.size == 10 * NUM_ANCHORS) {
            "Expected 21000 detector output values, got ${output.size}"
        }

        val candidates = mutableListOf<Candidate>()

        // Output shape: [1, 10, 2100]
        // Rows 0-3: cx, cy, width, height
        // Rows 4-9: six class probabilities

        for (anchor in 0 until NUM_ANCHORS) {

            var bestClass = 0
            var bestConfidence = Float.NEGATIVE_INFINITY

            for (classIndex in 0 until NUM_CLASSES) {

                val score = output[
                    (4 + classIndex) * NUM_ANCHORS + anchor
                ]

                if (!score.isFinite()) continue

                if (score > bestConfidence) {
                    bestConfidence = score
                    bestClass = classIndex
                }
            }

            if (bestConfidence < DECODE_CONFIDENCE) continue

            val cx = output[anchor] * IMAGE_SIZE
            val cy = output[NUM_ANCHORS + anchor] * IMAGE_SIZE
            val width = output[2 * NUM_ANCHORS + anchor] * IMAGE_SIZE
            val height = output[3 * NUM_ANCHORS + anchor] * IMAGE_SIZE

            if (!cx.isFinite() || !cy.isFinite() ||
                !width.isFinite() || !height.isFinite()
            ) continue

            if (width <= 0f || height <= 0f) continue

            candidates.add(
                Candidate(
                    anchorIndex = anchor,
                    classIndex = bestClass,
                    confidence = bestConfidence,
                    x1 = cx - width / 2f,
                    y1 = cy - height / 2f,
                    x2 = cx + width / 2f,
                    y2 = cy + height / 2f
                )
            )
        }

        // Class-aware NMS: suppress overlapping boxes only
        // when they belong to the same class.
        val kept = mutableListOf<Candidate>()

        for (classIndex in 0 until NUM_CLASSES) {

            val remaining = candidates
                .filter { it.classIndex == classIndex }
                .sortedWith(
                    compareByDescending<Candidate> { it.confidence }
                        .thenBy { it.anchorIndex }
                )
                .toMutableList()

            while (remaining.isNotEmpty()) {

                val best = remaining.removeAt(0)
                kept.add(best)

                val iterator = remaining.iterator()

                while (iterator.hasNext()) {
                    val candidate = iterator.next()

                    if (calculateIoU(best, candidate) > NMS_IOU) {
                        iterator.remove()
                    }
                }
            }
        }

        // Sort globally and retain at most 300 detections.
        return kept
            .sortedWith(
                compareByDescending<Candidate> { it.confidence }
                    .thenBy { it.classIndex }
                    .thenBy { it.anchorIndex }
            )
            .take(MAX_DETECTIONS)
            .map { candidate ->

                // Undo letterboxing and clip to original image bounds.
                val x1 = (
                    (candidate.x1 - transform.padX) / transform.scale
                ).coerceIn(0f, transform.originalWidth.toFloat())

                val y1 = (
                    (candidate.y1 - transform.padY) / transform.scale
                ).coerceIn(0f, transform.originalHeight.toFloat())

                val x2 = (
                    (candidate.x2 - transform.padX) / transform.scale
                ).coerceIn(0f, transform.originalWidth.toFloat())

                val y2 = (
                    (candidate.y2 - transform.padY) / transform.scale
                ).coerceIn(0f, transform.originalHeight.toFloat())

                VisionBox(
                    className = classNames[candidate.classIndex],
                    classIndex = candidate.classIndex,
                    confidence = candidate.confidence,
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2
                )
            }
    }

    private fun calculateIoU(
        a: Candidate,
        b: Candidate
    ): Float {

        val intersectionX1 = max(a.x1, b.x1)
        val intersectionY1 = max(a.y1, b.y1)
        val intersectionX2 = min(a.x2, b.x2)
        val intersectionY2 = min(a.y2, b.y2)

        val intersectionWidth =
            max(0f, intersectionX2 - intersectionX1)

        val intersectionHeight =
            max(0f, intersectionY2 - intersectionY1)

        val intersection =
            intersectionWidth * intersectionHeight

        val areaA =
            (a.x2 - a.x1) * (a.y2 - a.y1)

        val areaB =
            (b.x2 - b.x1) * (b.y2 - b.y1)

        val union = areaA + areaB - intersection

        return if (union <= 0f) {
            0f
        } else {
            intersection / union
        }
    }
}
