package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import kotlin.math.pow

class PoseEstimator(private val K: Mat) {

    companion object {
        private const val TAG = "PoseEstimator"
        private const val MIN_INLIERS = 15
    }

    fun estimatePose(matchSet: MatchSet): Pair<Mat, Mat> {

        val (pts1List, pts2List) = matchSet.getMatchedPoints()

        if (pts1List.size < MIN_INLIERS) {
            Log.w(TAG, "Not enough matches for essential matrix: ${pts1List.size}")
            return identity()
        }

        val pts1 = MatOfPoint2f(*pts1List.toTypedArray())
        val pts2 = MatOfPoint2f(*pts2List.toTypedArray())

        // FIX 1: Threshold at 2.0px — 1.0px was too strict for a handheld
        // phone camera with residual distortion and JPEG compression artifacts.
        // 2.0px is the practical sweet spot for mobile SfM.
        val E = Calib3d.findEssentialMat(
            pts1, pts2, K,
            Calib3d.RANSAC,
            0.999,
            2.0,
            1000
        )

        // FIX 2: Check E matrix shape BEFORE the empty check.
        // findEssentialMat can return multiple 3x3 matrices stacked vertically
        // (e.g. 6x3, 9x3). Passing a non-3x3 matrix to recoverPose produces
        // silent garbage poses without throwing any error.
        if (E.empty()) {
            Log.e(TAG, "Essential matrix estimation failed — empty result")
            return identity()
        }

        if (E.rows() % 3 != 0 || E.cols() != 3) {
            Log.e(TAG, "Bad Essential matrix shape: ${E.rows()}x${E.cols()}")
            return identity()
        }

        // Always take only the first 3x3 block
        val E33 = E.rowRange(0, 3)

        val R = Mat()
        val t = Mat()
        val poseMask = Mat()

        val recovered = Calib3d.recoverPose(E33, pts1, pts2, K, R, t, poseMask)

        Log.i(TAG, "recoverPose inliers: $recovered / ${pts1List.size}")

        if (recovered < MIN_INLIERS) {
            Log.w(TAG, "recoverPose found too few valid inliers: $recovered")
            return identity()
        }

        // Filter matchSet down to only pose-consistent inliers
        val final1 = ArrayList<Point>()
        val final2 = ArrayList<Point>()

        val nMask = minOf(pts1List.size, pts2List.size, poseMask.rows())
        for (i in 0 until nMask) {
            val mv = poseMask.get(i, 0)
            if (mv != null && mv.isNotEmpty() && mv[0] != 0.0) {
                final1.add(pts1List[i])
                final2.add(pts2List[i])
            }
        }

        if (final1.size < MIN_INLIERS) {
            Log.w(TAG, "Too few pose-consistent inliers after recoverPose: ${final1.size}")
            return identity()
        }

        matchSet.replaceMatches(final1, final2)

        // FIX 3: Log inlier ratio — useful for diagnosing match quality.
        // If this is consistently below 30%, the feature matcher is the bottleneck.
        val inlierRatio = final1.size.toFloat() / pts1List.size
        Log.i(TAG, "Pose estimated — matches: ${pts1List.size}, " +
                "inliers: ${final1.size}, ratio: ${"%.2f".format(inlierRatio)}")

        if (inlierRatio < 0.2f) {
            Log.w(TAG, "Low inlier ratio (${"%.2f".format(inlierRatio)}) " +
                    "— consider checking image overlap or exposure consistency")
        }

        val tMag = Math.sqrt(
            t.get(0,0)[0].pow(2) +
                    t.get(1,0)[0].pow(2) +
                    t.get(2,0)[0].pow(2)
        )
        if (tMag < 0.001) {
            Log.w(TAG, "recoverPose returned near-zero translation — returning identity fallback")
            return identity()
        }


        return R to t
    }

    // Clean helper — avoids repeating the identity/zero boilerplate
    private fun identity(): Pair<Mat, Mat> =
        Mat.eye(3, 3, CvType.CV_64F) to Mat.zeros(3, 1, CvType.CV_64F)

    // Used by the (commented out) homography path — kept for potential re-enable
    private fun countPointsInFront(
        K: Mat, R: Mat, t: Mat,
        pts1: MatOfPoint2f, pts2: MatOfPoint2f
    ): Int {
        val Rt1 = Mat(3, 4, CvType.CV_64F)
        for (r in 0 until 3) {
            for (c in 0 until 3) Rt1.put(r, c, if (r == c) 1.0 else 0.0)
            Rt1.put(r, 3, 0.0)
        }

        val R64 = Mat(); val t64 = Mat()
        R.convertTo(R64, CvType.CV_64F)
        t.convertTo(t64, CvType.CV_64F)

        val Rt2 = Mat(3, 4, CvType.CV_64F)
        for (r in 0 until 3) {
            for (c in 0 until 3) Rt2.put(r, c, R64.get(r, c)[0])
            Rt2.put(r, 3, t64.get(r, 0)[0])
        }

        val P1 = Mat(); val P2 = Mat()
        Core.gemm(K, Rt1, 1.0, Mat(), 0.0, P1)
        Core.gemm(K, Rt2, 1.0, Mat(), 0.0, P2)

        val pts4d = Mat()
        Calib3d.triangulatePoints(P1, P2, pts1, pts2, pts4d)

        var count = 0
        for (i in 0 until pts4d.cols()) {
            val w = pts4d.get(3, i)[0]
            if (w == 0.0) continue
            val X = pts4d.get(0, i)[0] / w
            val Y = pts4d.get(1, i)[0] / w
            val Z = pts4d.get(2, i)[0] / w

            val Xvec = Mat(3, 1, CvType.CV_64F)
            Xvec.put(0, 0, X); Xvec.put(1, 0, Y); Xvec.put(2, 0, Z)
            val X2 = Mat()
            Core.gemm(R64, Xvec, 1.0, t64, 1.0, X2)

            if (Z > 0 && X2.get(2, 0)[0] > 0) count++
        }
        return count
    }
}