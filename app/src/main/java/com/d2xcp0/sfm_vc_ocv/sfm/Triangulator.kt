package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.calib3d.Calib3d
import org.opencv.core.*

class Triangulator(private val K: Mat) {

    fun triangulate(matches: MatchSet, R: Mat, t: Mat): List<Point3> {

        val (pts1, pts2) = matches.getMatchedPoints()

        // Build P1 = K [I | 0]
        val P1 = Mat(3, 4, CvType.CV_64F)
        val I = Mat.eye(3, 3, CvType.CV_64F)
        val zero = Mat.zeros(3, 1, CvType.CV_64F)
        val left1 = Mat()
        Core.hconcat(listOf(I, zero), left1)
        Core.gemm(K, left1, 1.0, Mat(), 0.0, P1)

        // Build P2 = K [R | t]
        val Rt = Mat()
        Core.hconcat(listOf(R, t), Rt)
        val P2 = Mat()
        Core.gemm(K, Rt, 1.0, Mat(), 0.0, P2)

        // Convert matched points to OpenCV format
        val mat1 = MatOfPoint2f(*pts1.toTypedArray())
        val mat2 = MatOfPoint2f(*pts2.toTypedArray())

        // Perform triangulation
        val pts4d = Mat()
        Calib3d.triangulatePoints(P1, P2, mat1, mat2, pts4d)

        val cloud = mutableListOf<Point3>()
        for (i in 0 until pts4d.cols()) {
            val x = pts4d.get(0, i)[0] / pts4d.get(3, i)[0]
            val y = pts4d.get(1, i)[0] / pts4d.get(3, i)[0]
            val z = pts4d.get(2, i)[0] / pts4d.get(3, i)[0]
            cloud.add(Point3(x, y, z))   // ← FIXED: Point3, not Point3d
        }

        return cloud
    }
}
