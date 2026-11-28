package com.d2xcp0.sfm_vc_ocv.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.d2xcp0.sfm_vc_ocv.R

@Composable
fun MainScreen(
    onOpenGallery: () -> Unit,
    onOpenCamera: () -> Unit,
    onRunSfM: () -> Unit,
    onShowSfMResult: () -> Unit,
    onClearGallery: () -> Unit,
    onExportPointCloud: () -> Unit,
    onOpenDebug: () -> Unit,
    onSaveSession: () -> Unit,
    onLoadSession: () -> Unit,
    onLoadTestImages: () -> Unit
) {
    Scaffold(
        topBar = { AppTopBar(title = stringResource(R.string.app_name)) }
    ) { padding ->
        // Gorgetheto, hogy kis kijelzon se loghasson ki semmi
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            SectionTitle("Pictures")
            MainButton("Open Camera", onOpenCamera)
            MainButton("Open Gallery", onOpenGallery)
            MainButton("Clear Gallery", onClearGallery)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                MainButton("Save Session", onSaveSession, Modifier.weight(1f))
                MainButton("Load Session", onLoadSession, Modifier.weight(1f))
            }
            // assets/test_images: a beepitett S1 tesztsorozat (F1 teszt, ablacios meresek)
            MainButton("Load Test Images", onLoadTestImages)

            SectionTitle("Reconstruction")
            MainButton("Run SfM", onRunSfM)
            MainButton("View Reconstruction", onShowSfMResult)
            MainButton("Export Point Cloud", onExportPointCloud)

            SectionTitle("Helper")
            MainButton("Show Debug Matches", onOpenDebug)
        }
    }
}

@Composable
private fun SectionTitle(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.titleMedium,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier.padding(top = 12.dp)
    )
}

@Composable
private fun MainButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier.fillMaxWidth()
) {
    Button(onClick = onClick, modifier = modifier) { Text(text) }
}
