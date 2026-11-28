package com.d2xcp0.sfm_vc_ocv.screens

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import android.graphics.BitmapFactory
import android.graphics.Matrix
import java.io.File

private fun loadRotated(file: File): Bitmap? {
    val bmp = BitmapFactory.decodeFile(file.absolutePath) ?: return null
    val matrix = Matrix().apply { postRotate(90f) }
    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
}
@Composable
fun DebugGalleryScreen(
    debugFiles: List<File>,
    onBack: () -> Unit,
    onClearDebug: () -> Unit
) {
    // A rendszer vissza gombja/gesztusa a fokepernyore visz
    BackHandler(onBack = onBack)

    Scaffold(
        topBar = {
            AppTopBar(
                title = "Debug Match Images",
                onBack = onBack,
                actions = {
                    Button(
                        onClick = onClearDebug,
                        modifier = Modifier.padding(end = 8.dp)
                    ) {
                        Text("Clear All")
                    }
                }
            )
        }
    ) { padding ->
        LazyColumn(
            contentPadding = padding,
            modifier = Modifier.fillMaxSize()
        ) {
            items(debugFiles.size) { idx ->
                val file = debugFiles[idx]
                val bmp = BitmapFactory.decodeFile(file.absolutePath)
                //val bmp = loadRotated(file)

                Column(modifier = Modifier.padding(8.dp)) {
                    Text(text = file.name)
                    Spacer(modifier = Modifier.height(4.dp))

                    if (bmp != null) {
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = null,
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(4.dp)
                        )
                    }
                }
            }
        }
    }
}
