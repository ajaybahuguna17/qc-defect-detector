package com.example.executorchllamademo.aanetra

import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.executorchllamademo.aanetra.InspectionViewModel.AdviceState
import com.example.executorchllamademo.aanetra.InspectionViewModel.VisionStatus
import com.team.defectqc.core.QualityAction
import kotlinx.coroutines.delay
import java.util.Locale

@Composable
fun AANetraScreen(
    visionStatus: VisionStatus,
    modelState: ModelManager.State,
    busy: Boolean,
    result: InspectionViewModel.InspectionResult?,
    error: String?,
    advice: AdviceState,
    onPickImage: () -> Unit,
    onTakePhoto: () -> Unit,
    onRetryModel: () -> Unit,
    onRetryAdvice: () -> Unit,
    onNewInspection: () -> Unit,
    onAcceptMobileData: () -> Unit,
    onDeclineMobileData: () -> Unit,
    onOpenDeveloperSettings: () -> Unit
) {
    SteelBackground {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Header(onOpenDeveloperSettings)
            StatusCard(visionStatus, modelState, onRetryModel)

            val visionReady = visionStatus is VisionStatus.Ready
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                ActionButton(
                    text = "Pick image",
                    icon = { Icon(Icons.Filled.PhotoLibrary, contentDescription = null) },
                    enabled = visionReady && !busy,
                    onClick = onPickImage,
                    modifier = Modifier.weight(1f)
                )
                ActionButton(
                    text = "Take photo",
                    icon = { Icon(Icons.Filled.CameraAlt, contentDescription = null) },
                    enabled = visionReady && !busy,
                    onClick = onTakePhoto,
                    modifier = Modifier.weight(1f)
                )
            }

            if (busy) {
                SteelCard {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(24.dp), color = Steel.Orange, strokeWidth = 3.dp)
                        Spacer(Modifier.width(12.dp))
                        Text("Inspecting image...", style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }

            error?.let {
                SteelCard(borderColor = Steel.Red) {
                    Text(it, color = Steel.Red, style = MaterialTheme.typography.bodyLarge)
                }
            }

            result?.let {
                if (it.decision.action == QualityAction.RETAKE) {
                    RetakeCard(it, enabled = visionReady && !busy, onTakePhoto = onTakePhoto, onPickImage = onPickImage)
                    ImageWithBoxes(it)
                } else {
                    DecisionCard(it)
                    ImageWithBoxes(it)
                    VisionDetailsCard(it)
                    AdviceCard(advice, onRetryAdvice)
                }
                OutlinedButton(
                    onClick = onNewInspection,
                    enabled = !busy,
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    shape = RoundedCornerShape(10.dp),
                    border = BorderStroke(1.dp, Steel.Orange)
                ) { Text("New inspection", color = Steel.Orange, style = MaterialTheme.typography.labelLarge) }
            }

            if (result == null && !busy) {
                SteelCard {
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        listOf(
                            "1. Pick or take a photo",
                            "2. See the decision",
                            "3. Read the AI advice"
                        ).forEach {
                            Text(it, style = MaterialTheme.typography.bodyLarge, color = Steel.TextDim)
                        }
                    }
                }
            }

            Text(
                "Built with Llama",
                style = MaterialTheme.typography.labelMedium,
                color = Steel.TextDim,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(top = 4.dp, bottom = 8.dp)
            )
        }
    }

    if (modelState is ModelManager.State.NeedsMobileDataConsent) {
        AlertDialog(
            onDismissRequest = onDeclineMobileData,
            title = { Text("Download Llama model?") },
            text = {
                Text(
                    "765 MB download, Wi-Fi recommended.\n\n" +
                        "This is a one-time download. After it finishes, AANetra works fully offline."
                )
            },
            confirmButton = { TextButton(onClick = onAcceptMobileData) { Text("Continue") } },
            dismissButton = { TextButton(onClick = onDeclineMobileData) { Text("Cancel") } }
        )
    }
}

@Composable
private fun Header(onOpenDeveloperSettings: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text("AANetra", style = MaterialTheme.typography.headlineLarge, color = Steel.Orange)
            Text(
                "On-device steel defect inspection",
                style = MaterialTheme.typography.bodyMedium,
                color = Steel.TextDim
            )
        }
        IconButton(onClick = onOpenDeveloperSettings) {
            Icon(Icons.Filled.Settings, contentDescription = "Developer settings", tint = Steel.TextDim)
        }
    }
}

@Composable
private fun SteelCard(
    modifier: Modifier = Modifier,
    borderColor: Color = Steel.CardBorder,
    content: @Composable () -> Unit
) {
    val shape = RoundedCornerShape(12.dp)
    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(shape)
            .background(Steel.cardBrush)
            .border(1.dp, borderColor, shape)
            .padding(14.dp)
    ) { content() }
}

@Composable
private fun ActionButton(
    text: String,
    icon: @Composable () -> Unit,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.height(60.dp),
        shape = RoundedCornerShape(10.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = Steel.Orange,
            contentColor = Color(0xFF1A1000),
            disabledContainerColor = Steel.Card,
            disabledContentColor = Steel.TextDim
        )
    ) {
        icon()
        Spacer(Modifier.width(8.dp))
        Text(text, style = MaterialTheme.typography.labelLarge)
    }
}

@Composable
private fun StatusDot(color: Color) {
    Box(Modifier.size(12.dp).clip(CircleShape).background(color))
}

@Composable
private fun StatusCard(visionStatus: VisionStatus, modelState: ModelManager.State, onRetry: () -> Unit) {
    SteelCard {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                val (color, text) = when (visionStatus) {
                    VisionStatus.Loading -> Steel.Yellow to "Vision loading..."
                    VisionStatus.Ready -> Steel.Green to "Vision ready"
                    is VisionStatus.Error -> Steel.Red to "Vision error: ${visionStatus.message}"
                }
                StatusDot(color)
                Spacer(Modifier.width(10.dp))
                Text(text, style = MaterialTheme.typography.titleMedium)
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                val (color, text) = when (modelState) {
                    is ModelManager.State.Ready -> Steel.Green to "Llama ready"
                    is ModelManager.State.Downloading -> Steel.Yellow to "Downloading Llama model"
                    is ModelManager.State.Verifying -> Steel.Yellow to "Verifying ${modelState.fileName}..."
                    ModelManager.State.Loading -> Steel.Yellow to "Loading Llama..."
                    ModelManager.State.Checking, ModelManager.State.Idle -> Steel.Yellow to "Checking Llama model..."
                    is ModelManager.State.NeedsMobileDataConsent -> Steel.Yellow to "Llama model download needed"
                    is ModelManager.State.Error -> Steel.Red to "Llama not available"
                }
                StatusDot(color)
                Spacer(Modifier.width(10.dp))
                Text(text, style = MaterialTheme.typography.titleMedium)
            }

            when (modelState) {
                is ModelManager.State.Downloading -> {
                    val pct = InspectionViewModel.percent(modelState.doneBytes, modelState.totalBytes)
                    LinearProgressIndicator(
                        progress = { pct / 100f },
                        modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)),
                        color = Steel.Orange,
                        trackColor = Steel.Charcoal
                    )
                    Text(
                        "${ModelManager.mb(modelState.doneBytes)} / ${ModelManager.mb(modelState.totalBytes)} MB  ($pct%)",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Steel.TextDim
                    )
                }
                is ModelManager.State.Verifying, ModelManager.State.Loading -> LinearProgressIndicator(
                    modifier = Modifier.fillMaxWidth().height(6.dp),
                    color = Steel.Orange,
                    trackColor = Steel.Charcoal
                )
                is ModelManager.State.Error -> {
                    Text(modelState.message, color = Steel.Red, style = MaterialTheme.typography.bodyMedium)
                    OutlinedButton(
                        onClick = onRetry,
                        border = BorderStroke(1.dp, Steel.Orange)
                    ) { Text("Retry", color = Steel.Orange) }
                }
                else -> Unit
            }
        }
    }
}

private fun severityColor(severity: String?): Color = when (severity?.lowercase()) {
    "low" -> Steel.Green
    "medium" -> Steel.Yellow
    "high" -> Steel.SevOrange
    "critical" -> Steel.Red
    else -> Steel.Grey
}

private fun actionColor(action: QualityAction, stopLine: Boolean): Color = when {
    stopLine -> Steel.Red
    action == QualityAction.PASS || action == QualityAction.ACCEPT -> Steel.Green
    action == QualityAction.REWORK -> Steel.Yellow
    action == QualityAction.REJECT -> Steel.SevOrange
    action == QualityAction.RETAKE -> Steel.RetakeBlue
    else -> Steel.Grey
}

/** RETAKE: not a usable photo of a bare steel surface. No severity, no advice. */
@Composable
private fun RetakeCard(
    r: InspectionViewModel.InspectionResult,
    enabled: Boolean,
    onTakePhoto: () -> Unit,
    onPickImage: () -> Unit
) {
    SteelCard(borderColor = Steel.RetakeBlue) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("DECISION", style = MaterialTheme.typography.labelMedium, color = Steel.TextDim)
            Text(
                r.decision.label,
                fontSize = 40.sp,
                fontWeight = FontWeight.Black,
                color = Steel.RetakeBlue,
                textAlign = TextAlign.Center
            )
            Text(
                r.decision.explanation,
                style = MaterialTheme.typography.titleMedium,
                color = Steel.Text,
                textAlign = TextAlign.Center
            )
            Text(
                r.event.reason,
                style = MaterialTheme.typography.bodyMedium,
                color = Steel.TextDim,
                textAlign = TextAlign.Center
            )
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(
                    onClick = onTakePhoto,
                    enabled = enabled,
                    modifier = Modifier.weight(1f).height(56.dp),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Steel.RetakeBlue,
                        contentColor = Color(0xFF0E1620),
                        disabledContainerColor = Steel.Card,
                        disabledContentColor = Steel.TextDim
                    )
                ) { Text("Take photo again", style = MaterialTheme.typography.labelLarge, textAlign = TextAlign.Center) }
                OutlinedButton(
                    onClick = onPickImage,
                    enabled = enabled,
                    modifier = Modifier.weight(1f).height(56.dp),
                    shape = RoundedCornerShape(10.dp),
                    border = BorderStroke(1.dp, Steel.RetakeBlue)
                ) {
                    Text(
                        "Pick another image",
                        color = Steel.RetakeBlue,
                        style = MaterialTheme.typography.labelLarge,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Composable
private fun DecisionCard(r: InspectionViewModel.InspectionResult) {
    val color = actionColor(r.decision.action, r.decision.stopLine)
    SteelCard(borderColor = color) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("DECISION", style = MaterialTheme.typography.labelMedium, color = Steel.TextDim)
            Text(
                r.decision.label,
                fontSize = 40.sp,
                fontWeight = FontWeight.Black,
                color = color,
                textAlign = TextAlign.Center
            )
            Text(
                r.decision.explanation,
                style = MaterialTheme.typography.bodyMedium,
                color = Steel.Text,
                textAlign = TextAlign.Center
            )
        }
    }
}

@Composable
private fun ImageWithBoxes(r: InspectionViewModel.InspectionResult) {
    val bmp = r.bitmap
    val image = remember(bmp) { bmp.asImageBitmap() }
    val finalClass = r.event.finalClass
    val labelPaint = remember {
        Paint().apply {
            isAntiAlias = true
            typeface = Typeface.DEFAULT_BOLD
            color = android.graphics.Color.BLACK
        }
    }
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(bmp.width.toFloat() / bmp.height.toFloat())
            .clip(RoundedCornerShape(10.dp))
            .border(1.dp, Steel.CardBorder, RoundedCornerShape(10.dp))
    ) {
        Image(
            bitmap = image,
            contentDescription = "Inspected image",
            contentScale = ContentScale.FillBounds,
            modifier = Modifier.fillMaxSize()
        )
        Canvas(Modifier.fillMaxSize()) {
            val sx = size.width / bmp.width
            val sy = size.height / bmp.height
            labelPaint.textSize = 13.dp.toPx()
            r.event.boxes.forEach { box ->
                val (x1, y1, x2, y2) = box.bboxXYXY
                val color = if (box.className == finalClass) Steel.Orange else Steel.Yellow
                val left = x1 * sx
                val top = y1 * sy
                drawRect(
                    color = color,
                    topLeft = Offset(left, top),
                    size = Size((x2 - x1) * sx, (y2 - y1) * sy),
                    style = Stroke(width = 3.dp.toPx())
                )
                val label = String.format(Locale.US, "%s %.2f", box.className, box.confidence)
                val textW = labelPaint.measureText(label)
                val textH = labelPaint.textSize
                val pad = 3.dp.toPx()
                val labelTop = if (top - textH - 2 * pad >= 0) top - textH - 2 * pad else top
                drawRect(
                    color = color,
                    topLeft = Offset(left, labelTop),
                    size = Size(textW + 2 * pad, textH + 2 * pad)
                )
                drawContext.canvas.nativeCanvas.drawText(
                    label, left + pad, labelTop + pad + textH * 0.85f, labelPaint
                )
            }
        }
    }
}

@Composable
private fun SeverityBadge(severity: String?) {
    val color = severityColor(severity)
    Box(
        Modifier
            .clip(RoundedCornerShape(50))
            .background(color)
            .padding(horizontal = 14.dp, vertical = 6.dp)
    ) {
        Text(
            (severity ?: "None").uppercase(Locale.US),
            color = if (color == Steel.Yellow || color == Steel.Green) Color(0xFF111111) else Color.White,
            fontWeight = FontWeight.Bold,
            fontSize = 15.sp
        )
    }
}

@Composable
private fun VisionDetailsCard(r: InspectionViewModel.InspectionResult) {
    val e = r.event
    SteelCard {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Defect type", style = MaterialTheme.typography.labelMedium, color = Steel.TextDim)
                    Text(
                        if (e.alert) e.finalClass else "No defect",
                        style = MaterialTheme.typography.titleLarge
                    )
                }
                if (e.alert) SeverityBadge(e.severityLevel)
            }
            val conf = e.classifierConfidence?.let { String.format(Locale.US, "%.1f%%", it * 100f) } ?: "n/a"
            DetailLine("Decided by", e.decidedBy)
            DetailLine("Classifier", "${e.classifierClass ?: "n/a"} ($conf)")
            DetailLine("Defects found", "${e.numBoxes}")
            DetailLine(
                "Vision time",
                "${r.visionMs} ms (detector ${e.detectorMs} ms, classifier ${e.classifierMs} ms)"
            )
        }
    }
}

@Composable
private fun DetailLine(label: String, value: String) {
    Row {
        Text("$label: ", style = MaterialTheme.typography.bodyMedium, color = Steel.TextDim)
        Text(value, style = MaterialTheme.typography.bodyMedium, color = Steel.Text)
    }
}

@Composable
private fun AdviceCard(advice: AdviceState, onRetryAdvice: () -> Unit) {
    SteelCard {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("ADVICE", style = MaterialTheme.typography.labelMedium, color = Steel.Orange)
            when (advice) {
                AdviceState.None -> Unit
                AdviceState.NoDefect -> Text("No defect - no advice needed", style = MaterialTheme.typography.bodyLarge)
                is AdviceState.Skipped -> Text(advice.reason, style = MaterialTheme.typography.bodyLarge, color = Steel.TextDim)
                is AdviceState.NotReady -> Text(advice.reason, style = MaterialTheme.typography.bodyLarge, color = Steel.TextDim)
                is AdviceState.Generating -> {
                    var now by remember { mutableLongStateOf(android.os.SystemClock.elapsedRealtime()) }
                    LaunchedEffect(advice.startedAtMs) {
                        while (true) {
                            now = android.os.SystemClock.elapsedRealtime()
                            delay(250)
                        }
                    }
                    val secs = ((now - advice.startedAtMs).coerceAtLeast(0)) / 1000
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(22.dp), color = Steel.Orange, strokeWidth = 3.dp)
                        Spacer(Modifier.width(12.dp))
                        Text("Generating advice... ${secs}s", style = MaterialTheme.typography.bodyLarge)
                    }
                }
                is AdviceState.Done -> {
                    val a = advice.advisory
                    AdviceSection("SUMMARY", a.summary)
                    AdviceSection("CAUSE", a.likelyCause)
                    AdviceSection("ACTION", a.recommendedAction)
                    val footer = if (a.cached) {
                        String.format(Locale.US, "Advice in %.1f s (cached)", a.llmMs / 1000f)
                    } else {
                        String.format(
                            Locale.US, "Advice in %.1f s  |  %.1f tokens/s",
                            a.llmMs / 1000f, a.tokensPerSecond
                        )
                    }
                    Text(footer, style = MaterialTheme.typography.labelMedium, color = Steel.TextDim)
                }
                AdviceState.Failed -> {
                    Text(
                        "Advice could not be generated",
                        style = MaterialTheme.typography.bodyLarge,
                        color = Steel.Red
                    )
                    OutlinedButton(
                        onClick = onRetryAdvice,
                        border = BorderStroke(1.dp, Steel.Orange)
                    ) { Text("Retry advice", color = Steel.Orange) }
                }
            }
            Text(
                "Advice explains the defect only; it never changes the decision.",
                style = MaterialTheme.typography.labelMedium,
                color = Steel.TextDim
            )
        }
    }
}

@Composable
private fun AdviceSection(title: String, body: String) {
    Column {
        Text(title, style = MaterialTheme.typography.labelLarge, color = Steel.Orange)
        Text(body, style = MaterialTheme.typography.bodyLarge)
    }
}
