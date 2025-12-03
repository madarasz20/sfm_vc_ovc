package com.d2xcp0.sfm_vc_ocv.utils

import android.content.Context
import java.io.File

object StorageUtils {

    /**
     * Returns all image file paths stored in:
     *   /Android/data/<package>/files/Pictures/
     *
     * This is needed for the SfM pipeline (OpenMVG/OpenMVS).
     */
    fun getAllImageFilePaths(context: Context): List<String> {
        val storageDir: File = context.getExternalFilesDir("Pictures") ?: return emptyList()
        val files = storageDir.listFiles() ?: return emptyList()

        return files
            .filter { it.isFile && it.extension.lowercase() in listOf("jpg", "jpeg", "png") }
            .sortedBy { it.lastModified() }
            .map { it.absolutePath }
    }

    fun loadDebugImages(context: Context): List<File> {
        val dir = File(context.getExternalFilesDir(null), "debug")
        if (!dir.exists()) return emptyList()

        return dir.listFiles()
            ?.filter { it.extension.lowercase() in listOf("jpg", "png") }
            ?.sortedBy { it.name }
            ?: emptyList()
    }
    fun clearDebugImages(context: Context) {
        val dir = File(context.getExternalFilesDir(null), "debug")
        if (!dir.exists()) return

        dir.listFiles()?.forEach { file ->
            try {
                file.delete()
            } catch (_: Exception) {}
        }
    }


}
