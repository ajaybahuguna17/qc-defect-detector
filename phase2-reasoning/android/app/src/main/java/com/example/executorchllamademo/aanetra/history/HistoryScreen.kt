package com.example.executorchllamademo.aanetra.history

import android.graphics.BitmapFactory
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.executorchllamademo.aanetra.Steel
import com.example.executorchllamademo.aanetra.SteelBackground
import com.example.executorchllamademo.aanetra.formatLocalDateTime
import com.example.executorchllamademo.aanetra.severityColor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

/** History list + detail. Back from the detail returns to the list, back from the list closes it. */
@Composable
fun HistoryRoute(
    records: List<InspectionRecord>,
    onBack: () -> Unit,
    onDeleteAll: () -> Unit,
    onExport: () -> Unit
) {
    var selectedId by rememberSaveable { mutableStateOf<Long?>(null) }
    val selected = selectedId?.let { id -> records.firstOrNull { it.id == id } }

    BackHandler { if (selectedId != null) selectedId = null else onBack() }

    SteelBackground {
        if (selected != null) {
            HistoryDetail(selected, onBack = { selectedId = null })
        } else {
            HistoryList(records, onBack, onOpen = { selectedId = it }, onDeleteAll = onDeleteAll, onExport = onExport)
        }
    }
}

internal fun decisionColor(decision: String): Color {
    val d = decision.uppercase(Locale.US)
    return when {
        d.contains("STOP LINE") -> Steel.Red
        d.startsWith("PASS") || d.startsWith("ACCEPT") -> Steel.Green
        d.startsWith("REWORK") -> Steel.Yellow
        d.startsWith("REJECT") -> Steel.SevOrange
        d.startsWith("RETAKE") -> Steel.RetakeBlue
        else -> Steel.Grey
    }
}

@Composable
private fun TopBar(title: String, onBack: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = onBack) {
            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Steel.Text)
        }
        Text(title, style = MaterialTheme.typography.headlineSmall, color = Steel.Orange)
    }
}

@Composable
private fun HistoryList(
    records: List<InspectionRecord>,
    onBack: () -> Unit,
    onOpen: (Long) -> Unit,
    onDeleteAll: () -> Unit,
    onExport: () -> Unit
) {
    var confirmDelete by remember { mutableStateOf(false) }
    Column(
        Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        TopBar("Inspection history", onBack)
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(
                onClick = onExport,
                enabled = records.isNotEmpty(),
                modifier = Modifier.weight(1f).height(48.dp),
                border = BorderStroke(1.dp, Steel.Orange)
            ) { Text("Export CSV", color = if (records.isNotEmpty()) Steel.Orange else Steel.TextDim) }
            OutlinedButton(
                onClick = { confirmDelete = true },
                enabled = records.isNotEmpty(),
                modifier = Modifier.weight(1f).height(48.dp),
                border = BorderStroke(1.dp, Steel.Red)
            ) { Text("Delete all", color = if (records.isNotEmpty()) Steel.Red else Steel.TextDim) }
        }
        Text("${records.size} inspection(s), newest first", style = MaterialTheme.typography.labelMedium, color = Steel.TextDim)
        if (records.isEmpty()) {
            Text("No inspections logged yet.", style = MaterialTheme.typography.bodyLarge, color = Steel.TextDim)
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            items(records, key = { it.id }) { r -> HistoryRow(r) { onOpen(r.id) } }
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete all inspections?") },
            text = { Text("This removes all ${records.size} logged inspections and their thumbnails from this phone. It cannot be undone.") },
            confirmButton = {
                TextButton(onClick = { confirmDelete = false; onDeleteAll() }) { Text("Delete all", color = Steel.Red) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } }
        )
    }
}

@Composable
private fun Thumbnail(path: String?, modifier: Modifier) {
    val bmp by produceState<ImageBitmap?>(null, path) {
        value = path?.let { p ->
            withContext(Dispatchers.IO) { runCatching { BitmapFactory.decodeFile(p)?.asImageBitmap() }.getOrNull() }
        }
    }
    Box(modifier.clip(RoundedCornerShape(6.dp)).background(Steel.Charcoal)) {
        bmp?.let {
            Image(it, contentDescription = "Inspection thumbnail", contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
        }
    }
}

@Composable
private fun Pill(text: String, color: Color) {
    Box(Modifier.clip(RoundedCornerShape(50)).background(color).padding(horizontal = 10.dp, vertical = 3.dp)) {
        Text(
            text,
            color = if (color == Steel.Yellow || color == Steel.Green) Color(0xFF111111) else Color.White,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun HistoryRow(r: InspectionRecord, onClick: () -> Unit) {
    val context = LocalContext.current
    val shape = RoundedCornerShape(10.dp)
    Row(
        Modifier.fillMaxWidth().clip(shape).background(Steel.cardBrush).border(1.dp, Steel.CardBorder, shape)
            .clickable(onClick = onClick).padding(10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Thumbnail(r.thumbnailPath, Modifier.size(64.dp))
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(formatLocalDateTime(context, r.timestampMs), style = MaterialTheme.typography.labelMedium, color = Steel.TextDim)
            Text(r.defectType, style = MaterialTheme.typography.titleMedium, color = Steel.Text)
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                r.severity?.let { Pill(it.uppercase(Locale.US), severityColor(it)) }
                Pill(r.decision, decisionColor(r.decision))
            }
        }
    }
}

@Composable
private fun HistoryDetail(r: InspectionRecord, onBack: () -> Unit) {
    val context = LocalContext.current
    Column(
        Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        TopBar("Inspection #${r.id}", onBack)
        Thumbnail(r.thumbnailPath, Modifier.fillMaxWidth().aspectRatio(1f))
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            r.severity?.let { Pill(it.uppercase(Locale.US), severityColor(it)) }
            Pill(r.decision, decisionColor(r.decision))
        }
        val conf = r.classifierConfidence?.let { String.format(Locale.US, "%.1f %%", it * 100f) } ?: "n/a"
        val fields = listOf(
            "Inspected" to formatLocalDateTime(context, r.timestampMs),
            "Source" to r.source,
            "Defect type" to r.defectType,
            "Severity" to (r.severity ?: "none"),
            "Decision" to r.decision,
            "Decided by" to r.decidedBy,
            "Classifier confidence" to conf,
            "Defects found" to r.defectsFound.toString(),
            "Vision time" to "${r.visionMs} ms",
            "RETAKE / HOLD reason" to (r.gateReason ?: "-"),
            "Advice time" to (r.adviceMs?.let { String.format(Locale.US, "%.1f s", it / 1000f) } ?: "-"),
            "Advice cached" to when (r.adviceCached) { true -> "yes"; false -> "no"; null -> "-" },
            "Advice" to (r.adviceText ?: "-")
        )
        fields.forEach { (k, v) ->
            Column {
                Text(k, style = MaterialTheme.typography.labelMedium, color = Steel.TextDim)
                Text(v, style = MaterialTheme.typography.bodyLarge, color = Steel.Text)
            }
        }
    }
}
