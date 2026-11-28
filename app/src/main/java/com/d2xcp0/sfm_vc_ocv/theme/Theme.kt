package com.d2xcp0.sfm_vc_ocv.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

// Egyetlen, rogzitett paletta: nincs dynamic color, es a sotet mod sem irja felul.
private val AppColorScheme = lightColorScheme(
    primary = Brown,
    onPrimary = Beige,
    primaryContainer = BeigeDark,
    onPrimaryContainer = DarkBrown,
    secondary = Brown,
    onSecondary = Beige,
    secondaryContainer = BeigeDark,
    onSecondaryContainer = DarkBrown,
    tertiary = Brown,
    onTertiary = Beige,
    background = Beige,
    onBackground = DarkBrown,
    surface = Beige,
    onSurface = DarkBrown,
    surfaceVariant = BeigeDark,
    onSurfaceVariant = DarkBrown,
    outline = Brown
)

@Composable
fun Sfm_vcTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = AppColorScheme,
        typography = Typography,
        content = content
    )
}
