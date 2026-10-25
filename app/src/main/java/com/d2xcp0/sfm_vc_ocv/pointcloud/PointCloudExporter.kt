package com.d2xcp0.sfm_vc_ocv.pointcloud

import android.content.Context
import org.opencv.core.Point3
import java.io.File

object PointCloudExporter {

    // colors: pontonkenti RGB 0..1 (points sorrendjeben); null eseten szin nelkuli PLY
    fun exportPLY(context: Context, points: List<Point3>, colors: List<FloatArray>? = null): File? {
        if (points.isEmpty()) return null

        val withColor = colors != null && colors.size == points.size

        val file = File(context.getExternalFilesDir(null), "pointcloud.ply")

        val builder = StringBuilder()
        builder.append("ply\n")
        builder.append("format ascii 1.0\n")
        builder.append("element vertex ${points.size}\n")
        builder.append("property float x\n")
        builder.append("property float y\n")
        builder.append("property float z\n")
        if (withColor) {
            builder.append("property uchar red\n")
            builder.append("property uchar green\n")
            builder.append("property uchar blue\n")
        }
        builder.append("end_header\n")

        for (i in points.indices) {
            val p = points[i]
            builder.append("${p.x} ${p.y} ${p.z}")
            if (withColor) {
                val c = colors!![i]
                builder.append(" ${toByte(c[0])} ${toByte(c[1])} ${toByte(c[2])}")
            }
            builder.append("\n")
        }

        file.writeText(builder.toString())

        return file
    }

    private fun toByte(v: Float): Int = (v * 255f).toInt().coerceIn(0, 255)
}
