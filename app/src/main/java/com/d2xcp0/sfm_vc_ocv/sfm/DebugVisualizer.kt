package com.d2xcp0.sfm_vc_ocv.sfm

import android.content.Context
import org.opencv.core.*
import org.opencv.features2d.Features2d
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

object DebugVisualizer {

    fun saveMatchesImage(
        context: Context,
        img1: Mat,
        img2: Mat,
        kp1: MatOfKeyPoint,
        kp2: MatOfKeyPoint,
        matches: List<Pair<Point, Point>>,
        name: String
    ) {
        try {
            val outImg = Mat()

            // Convert matches to DMatch format
            val dmatches = ArrayList<DMatch>()
            for ((idx, _) in matches.withIndex()) {
                dmatches.add(DMatch(idx, idx, 1f))
            }

            val m = MatOfDMatch()
            m.fromList(dmatches)

            Features2d.drawMatches(
                img1, kp1,
                img2, kp2,
                m,
                outImg
            )

            val dir = File(context.getExternalFilesDir(null), "debug")
            if (!dir.exists()) dir.mkdirs()

            val file = File(dir, "$name.jpg")
            Imgcodecs.imwrite(file.absolutePath, outImg)

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
