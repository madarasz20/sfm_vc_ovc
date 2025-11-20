package com.d2xcp0.sfm_vc_ocv.sfm

import android.content.Context
import org.opencv.core.Point3
import java.io.File

object PointCloudExporter {

    fun exportPLY(context: Context, points: List<Point3>): File? {
        if (points.isEmpty()) return null

        val file = File(context.getExternalFilesDir(null), "pointcloud.ply")

        val builder = StringBuilder()
        builder.append("ply\n")
        builder.append("format ascii 1.0\n")
        builder.append("element vertex ${points.size}\n")
        builder.append("property float x\n")
        builder.append("property float y\n")
        builder.append("property float z\n")
        builder.append("end_header\n")

        for (p in points) {
            builder.append("${p.x} ${p.y} ${p.z}\n")
        }

        file.writeText(builder.toString())

        return file
    }
}
