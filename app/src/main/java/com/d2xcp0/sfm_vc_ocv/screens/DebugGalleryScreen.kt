package com.d2xcp0.sfm_vc_ocv.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import android.graphics.BitmapFactory
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import java.io.File

@Composable
fun DebugGalleryScreen(
    debugFiles: List<File>,
    onBack: () -> Unit,
    onClearDebug: () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Debug Match Images") },
                navigationIcon = {
                IconButton(onClick = onBack) {
                    Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                }
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
            },
                actions = {
                    Button(onClick = onClearDebug) {
                        Text("Clear All")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors())
        }
    ) { padding ->
        LazyColumn(
            contentPadding = padding,
            modifier = Modifier.fillMaxSize()
        ) {
            items(debugFiles.size) { idx ->
                val file = debugFiles[idx]
                val bmp = BitmapFactory.decodeFile(file.absolutePath)

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
