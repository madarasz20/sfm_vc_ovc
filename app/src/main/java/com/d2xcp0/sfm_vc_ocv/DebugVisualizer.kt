package com.d2xcp0.sfm_vc_ocv

import android.content.Context
import org.opencv.core.*
import org.opencv.features2d.Features2d
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
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
            // Rotate images 90° for display — source images are 1440x1080 landscape
            // but phone is held portrait, so rotate before stitching the composite
            val rot1 = rotateMat90(img1)
            val rot2 = rotateMat90(img2)

            // Keypoints also need to be transformed to match rotated image coordinates
            val rkp1 = rotateKeypoints(kp1, img1.cols(), img1.rows())
            val rkp2 = rotateKeypoints(kp2, img2.cols(), img2.rows())

            val outImg = Mat()
            val dmatches = ArrayList<DMatch>()
            for ((idx, _) in matches.withIndex()) {
                dmatches.add(DMatch(idx, idx, 1f))
            }

            val m = MatOfDMatch()
            m.fromList(dmatches)

            Features2d.drawMatches(rot1, rkp1, rot2, rkp2, m, outImg)

            val dir = File(context.getExternalFilesDir(null), "debug")
            if (!dir.exists()) dir.mkdirs()

            val file = File(dir, "$name.jpg")
            Imgcodecs.imwrite(file.absolutePath, outImg)

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // Rotate Mat 90° clockwise: (x, y) in WxH → (H-1-y, x) in HxW
    private fun rotateMat90(src: Mat): Mat {
        val dst = Mat()
        Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE)
        return dst
    }

    // Transform keypoint coordinates for 90° clockwise rotation
    // Original: (x, y) in image of width W, height H
    // Rotated:  (H - 1 - y, x) in image of width H, height W
    private fun rotateKeypoints(kp: MatOfKeyPoint, srcWidth: Int, srcHeight: Int): MatOfKeyPoint {
        val original = kp.toArray()
        val rotated = original.map { keypoint ->
            val newX = (srcHeight - 1 - keypoint.pt.y).toFloat()
            val newY = keypoint.pt.x.toFloat()
            org.opencv.core.KeyPoint(
                newX, newY,
                keypoint.size,
                keypoint.angle,
                keypoint.response,
                keypoint.octave,
                keypoint.class_id
            )
        }
        val result = MatOfKeyPoint()
        result.fromList(rotated)
        return result
    }
}