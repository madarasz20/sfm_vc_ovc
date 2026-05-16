package com.d2xcp0.sfm_vc_ocv.sfm

import android.net.Uri
import android.view.TextureView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView

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

    Column(
        modifier = Modifier.fillMaxSize()
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

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Button(onClick = onBack) {
                Text("Back")
            }

            Button(
                onClick = {
                    cameraManager.capturePhoto { uri ->
                        onPhotoCaptured(uri)
                    }
                }
            ) {
                Text("Capture")
            }
        }
    }
}