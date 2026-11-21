package com.example.executorchllamademo.aanetra

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/** Dark industrial palette, all drawn in code (no image assets). */
object Steel {
    val Charcoal = Color(0xFF16181B)
    val CharcoalLight = Color(0xFF24282D)
    val Card = Color(0xFF30363C)
    val CardBorder = Color(0xFF4A535C)
    val Orange = Color(0xFFFF8A1F)
    val OrangeDark = Color(0xFFB85C0A)
    val Text = Color(0xFFECEFF1)
    val TextDim = Color(0xFFA7B0B8)

    val Green = Color(0xFF3FAF5A)
    val Yellow = Color(0xFFF2C230)
    val SevOrange = Color(0xFFFF8A1F)
    val Red = Color(0xFFE53935)
    val Grey = Color(0xFF8A939B)
    /** RETAKE (not a usable steel photo): grey-blue, neutral, not a part decision. */
    val RetakeBlue = Color(0xFF7A9CB8)

    /** Vertical brushed-steel gradient: soft bands of slightly lighter/darker charcoal. */
    val backgroundBrush = Brush.verticalGradient(
        0.00f to Color(0xFF2A2E33),
        0.18f to Color(0xFF1C1F23),
        0.40f to Color(0xFF23272B),
        0.62f to Color(0xFF17191C),
        0.85f to Color(0xFF202327),
        1.00f to Color(0xFF141619)
    )

    val cardBrush = Brush.verticalGradient(
        listOf(Color(0xFF3A4148), Color(0xFF2D3338))
    )
}

private val AANetraColors = darkColorScheme(
    primary = Steel.Orange,
    onPrimary = Color(0xFF1A1000),
    secondary = Steel.Grey,
    background = Steel.Charcoal,
    onBackground = Steel.Text,
    surface = Steel.Card,
    onSurface = Steel.Text,
    surfaceVariant = Steel.CharcoalLight,
    onSurfaceVariant = Steel.TextDim,
    error = Steel.Red
)

private val AANetraType = Typography(
    headlineLarge = TextStyle(fontSize = 34.sp, fontWeight = FontWeight.Black, letterSpacing = 1.sp),
    titleLarge = TextStyle(fontSize = 22.sp, fontWeight = FontWeight.Bold),
    titleMedium = TextStyle(fontSize = 18.sp, fontWeight = FontWeight.SemiBold),
    bodyLarge = TextStyle(fontSize = 18.sp),
    bodyMedium = TextStyle(fontSize = 16.sp),
    labelLarge = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.SemiBold),
    labelMedium = TextStyle(fontSize = 14.sp)
)

@Composable
fun AANetraTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = AANetraColors, typography = AANetraType, content = content)
}

/** Full-screen brushed-steel background with fine horizontal brush lines. */
@Composable
fun SteelBackground(content: @Composable BoxScope.() -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Steel.backgroundBrush)
            .drawBehind {
                var y = 0f
                var i = 0
                val step = 3f
                while (y < size.height) {
                    val alpha = if (i % 2 == 0) 0.025f else 0.012f
                    drawLine(
                        color = if (i % 3 == 0) Color.White.copy(alpha = alpha) else Color.Black.copy(alpha = alpha),
                        start = Offset(0f, y),
                        end = Offset(size.width, y),
                        strokeWidth = 1f
                    )
                    y += step
                    i++
                }
            },
        content = content
    )
}
