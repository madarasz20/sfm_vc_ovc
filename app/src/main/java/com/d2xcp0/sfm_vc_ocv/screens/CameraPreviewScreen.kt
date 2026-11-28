package com.d2xcp0.sfm_vc_ocv.screens

import android.net.Uri
import android.view.TextureView
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.d2xcp0.sfm_vc_ocv.camera.Camera2CaptureManager

@Composable
fun CameraPreviewScreen(
    onPhotoCaptured: (Uri) -> Unit,
    onBack: () -> Unit
) {
    val context = LocalContext.current

    val cameraManager = remember {
        Camera2CaptureManager(context)
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraManager.closeCamera()
        }
    }

    // A rendszer vissza gombja/gesztusa a fokepernyore visz
    BackHandler(onBack = onBack)

    Scaffold(
        topBar = { AppTopBar(title = "Camera", onBack = onBack) }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            AndroidView(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                factory = { ctx ->
                    TextureView(ctx).also { textureView ->
                        cameraManager.startCamera(textureView)
                    }
                }
            )

            Button(
                onClick = {
                    cameraManager.capturePhoto { uri ->
                        onPhotoCaptured(uri)
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Text("Capture")
            }
        }
    }
}
