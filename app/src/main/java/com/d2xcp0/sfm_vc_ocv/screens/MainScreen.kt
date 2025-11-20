package com.d2xcp0.sfm_vc_ocv.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun MainScreen(
    onOpenGallery: () -> Unit,
    onOpenCamera: () -> Unit,
    onTestImagePaths: () -> Unit,
    onRunSfM: () -> Unit,
    onShowSfMResult: () -> Unit,
    onClearGallery: () -> Unit,
    onExportPointCloud: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        Button(onClick = onOpenCamera) { Text("Open Camera") }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onOpenGallery) { Text("Open Gallery") }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onTestImagePaths) { Text("Test Image Paths") }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onRunSfM) { Text("Run SfM") }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onShowSfMResult) { Text("View Reconstruction (Later)") }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onClearGallery) {
            Text("Clear Gallery")       // NEW
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onExportPointCloud) {
            Text("Export Point Cloud")
        }
    }
}
