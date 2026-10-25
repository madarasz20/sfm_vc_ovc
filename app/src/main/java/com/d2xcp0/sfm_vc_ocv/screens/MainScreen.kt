package com.d2xcp0.sfm_vc_ocv.screens


import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Alignment.Companion.CenterStart
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.graphics.Color

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
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.Start
    ) {

        Button(onClick = onOpenCamera,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) { Text("Open Camera")}
        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onOpenGallery,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) { Text("Open Gallery") }
        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onClearGallery,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) { Text("Clear Gallery") }
        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onRunSfM,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) { Text("Run SfM") }
        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onShowSfMResult,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) { Text("View Reconstruction") }
        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = onOpenDebug,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) { Text("Show Debug Matches") }
        Spacer(Modifier.height(16.dp))

        Button(onClick = onExportPointCloud,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) { Text("Export Point Cloud") }
        Spacer(Modifier.height(16.dp))

        Button(onClick = onSaveSession,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) {Text("Save Session")}
        Spacer(Modifier.height(16.dp))

        Button(onClick = onLoadSession,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) {Text("Load Session")}
        Spacer(Modifier.height(16.dp))

        // ideiglenes: assets/test_images kepei teszteleshez
        Button(onClick = onLoadTestImages,
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.LightGray,
                contentColor = Color.Black
            )

        ) {Text("Load Test Images")}

    }
}
