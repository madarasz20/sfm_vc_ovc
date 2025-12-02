package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*

class PoseEstimator(private val K: Mat) {

    companion object {
        private const val TAG = "PoseEstimator"
        private const val MIN_INLIERS = 12
    }

    fun estimatePose(matchSet: MatchSet): Pair<Mat, Mat> {

        val (pts1List, pts2List) = matchSet.getMatchedPoints()

        if (pts1List.size < MIN_INLIERS) {
            Log.w(TAG, "Not enough matches for homography")
            return Mat.eye(3,3, CvType.CV_64F) to Mat.zeros(3,1, CvType.CV_64F)
        }

        val pts1 = MatOfPoint2f(*pts1List.toTypedArray())
        val pts2 = MatOfPoint2f(*pts2List.toTypedArray())

        // --- 1) Homography with RANSAC ---
        val mask = Mat()
        val H = Calib3d.findHomography(
            pts1,
            pts2,
            Calib3d.RANSAC,
            3.0,
            mask
        )

        if (H.empty()) {
            Log.e(TAG, "Homography failed")
            return Mat.eye(3,3, CvType.CV_64F) to Mat.zeros(3,1, CvType.CV_64F)
        }

        // keep only inliers for decomposition / triangulation
        val in1 = ArrayList<Point>()
        val in2 = ArrayList<Point>()
        for (i in pts1List.indices) {
            if (mask.get(i, 0)[0] > 0) {
                in1.add(pts1List[i])
                in2.add(pts2List[i])
            }
        }

        if (in1.size < MIN_INLIERS) {
            Log.w(TAG, "Too few inliers after RANSAC")
            return Mat.eye(3,3, CvType.CV_64F) to Mat.zeros(3,1, CvType.CV_64F)
        }

        matchSet.replaceMatches(in1, in2)

        val inPts1 = MatOfPoint2f(*in1.toTypedArray())
        val inPts2 = MatOfPoint2f(*in2.toTypedArray())

        // --- 2) Decompose H ---
        val rotations = ArrayList<Mat>()
        val translations = ArrayList<Mat>()
        val normals = ArrayList<Mat>()
        val nSol = Calib3d.decomposeHomographyMat(H, K, rotations, translations, normals)

        if (nSol == 0) {
            Log.e(TAG, "No homography decomposition solutions")
            return Mat.eye(3,3, CvType.CV_64F) to Mat.zeros(3,1, CvType.CV_64F)
        }

        // --- 3) Cheirality test: choose solution with most points in front of both cameras ---
        var bestIdx = 0
        var bestCount = -1

        for (i in 0 until nSol) {
            val R = rotations[i]
            val t = translations[i]

            val count = countPointsInFront(K, R, t, inPts1, inPts2)
            Log.i(TAG, "Solution $i has $count points with positive depth")

            if (count > bestCount) {
                bestCount = count
                bestIdx = i
            }
        }

        if (bestCount < 5) {
            Log.w(TAG, "No good cheirality solution, using identity pose")
            return Mat.eye(3,3, CvType.CV_64F) to Mat.zeros(3,1, CvType.CV_64F)
        }

        val Rbest = rotations[bestIdx]
        val tbest = translations[bestIdx]

        Log.i(TAG, "Selected solution $bestIdx with $bestCount in-front points")

        return Rbest to tbest
    }

    // Count how many triangulated points are in front of both cameras
    private fun countPointsInFront(
        K: Mat,
        R: Mat,
        t: Mat,
        pts1: MatOfPoint2f,
        pts2: MatOfPoint2f
    ): Int {

        // Camera 1: [I | 0]
        val Rt1 = Mat(3, 4, CvType.CV_64F)
        for (r in 0 until 3) {
            for (c in 0 until 3) {
                Rt1.put(r, c, if (r == c) 1.0 else 0.0)
            }
            Rt1.put(r, 3, 0.0)
        }

        // Camera 2: [R | t]
        val R64 = Mat()
        val t64 = Mat()
        R.convertTo(R64, CvType.CV_64F)
        t.convertTo(t64, CvType.CV_64F)

        val Rt2 = Mat(3, 4, CvType.CV_64F)
        for (r in 0 until 3) {
            for (c in 0 until 3) {
                Rt2.put(r, c, R64.get(r, c)[0])
            }
            Rt2.put(r, 3, t64.get(r, 0)[0])
        }

        val P1 = Mat()
        val P2 = Mat()
        Core.gemm(K, Rt1, 1.0, Mat(), 0.0, P1)
        Core.gemm(K, Rt2, 1.0, Mat(), 0.0, P2)

        val pts4d = Mat()
        Calib3d.triangulatePoints(P1, P2, pts1, pts2, pts4d)

        var count = 0
        val n = pts4d.cols()
        for (i in 0 until n) {
            val x = pts4d.get(0, i)[0]
            val y = pts4d.get(1, i)[0]
            val z = pts4d.get(2, i)[0]
            val w = pts4d.get(3, i)[0]
            if (w == 0.0) continue

            val X = x / w
            val Y = y / w
            val Z = z / w

            // depth in cam1
            val Z1 = Z

            // depth in cam2: R*X + t
            val Xvec = Mat(3, 1, CvType.CV_64F)
            Xvec.put(0, 0, X)
            Xvec.put(1, 0, Y)
            Xvec.put(2, 0, Z)
            val X2 = Mat()
            Core.gemm(R64, Xvec, 1.0, t64, 1.0, X2)
            val Z2 = X2.get(2, 0)[0]

            if (Z1 > 0 && Z2 > 0) {
                count++
            }
        }

        return count
    }
}
