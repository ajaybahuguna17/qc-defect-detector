
package com.team.defectqc.vision

import com.team.defectqc.core.DefectBox
import com.team.defectqc.core.DetectionEvent
import java.util.UUID

object VisionResultMapper {

    fun toDetectionEvent(
        result: VisionResult,
        imageName: String
    ): DetectionEvent {

        val finalBoxes = result.boxes.filter {
            it.className == result.finalClass
        }

        return DetectionEvent(
            id = UUID.randomUUID().toString(),
            timestampEpochMs = System.currentTimeMillis(),

            imageName = imageName,
            detector = result.detector,

            alert = result.alert,
            finalClass = result.finalClass,
            decidedBy = result.decidedBy,

            classifierClass = result.classifier.className,
            classifierConfidence = result.classifier.confidence,

            numBoxes = result.boxes.size,
            numFinalBoxes = finalBoxes.size,
            largestAreaPct = result.largestAreaPct,

            severityLevel = result.severityLevel,
            reason = result.reason,

            boxes = result.boxes.map { box ->
                DefectBox(
                    className = box.className,
                    confidence = box.confidence,
                    bboxXYXY = listOf(
                        box.x1,
                        box.y1,
                        box.x2,
                        box.y2
                    )
                )
            },

            detectorMs = result.detectorMs,
            classifierMs = result.classifierMs
        )
    }
}
