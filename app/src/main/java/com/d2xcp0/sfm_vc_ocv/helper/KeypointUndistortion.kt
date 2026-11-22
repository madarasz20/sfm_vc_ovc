package com.d2xcp0.sfm_vc_ocv.helper

import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f

// Kulon objektumban, hogy unit tesztbol is hivhato legyen (korabban a MainActivity privat fuggvenye volt)
object KeypointUndistortion {

    // Kulcspontok torzitasmentesitese pixel koordinatakban (P = K).
    // A sorrend (es igy a descriptor indexek) nem valtozik.
    fun undistortKeypoints(kps: MatOfKeyPoint, K: Mat, D: Mat): MatOfKeyPoint {
        val arr = kps.toArray()
        if (arr.isEmpty()) return kps

        val src = MatOfPoint2f(*arr.map { it.pt }.toTypedArray())
        val dst = MatOfPoint2f()
        Calib3d.undistortPoints(src, dst, K, D, Mat(), K)

        val pts = dst.toArray()
        for (i in arr.indices) arr[i].pt = pts[i]
        return MatOfKeyPoint(*arr)
    }
}
