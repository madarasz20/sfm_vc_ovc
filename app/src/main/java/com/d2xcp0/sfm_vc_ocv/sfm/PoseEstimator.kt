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

        val originalIndexedMatches = matchSet.getIndexedMatches()

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

        val eMask = Mat()

        val E = Calib3d.findEssentialMat(
            pts1, pts2, K,
            Calib3d.RANSAC,//LMEDS
            0.999,
            1.0,        //2.0
            1000,
            eMask
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
        val poseMask = eMask.clone()

        val recovered = Calib3d.recoverPose(
            E33,
            pts1,
            pts2,
            K,
            R,
            t,
            poseMask
        )

        val eInliers = Core.countNonZero(eMask)

        Log.i(TAG, "findEssentialMat inliers: $eInliers / ${pts1List.size}")
        Log.i(TAG, "recoverPose inliers: $recovered / ${pts1List.size}")
        Log.i(
            TAG,
            "recoverPose output: " +
                    "R=${R.rows()}x${R.cols()} empty=${R.empty()}, " +
                    "t=${t.rows()}x${t.cols()} empty=${t.empty()}"
        )

        if (recovered < MIN_INLIERS) {
            Log.w(TAG, "recoverPose found too few valid inliers: $recovered")
            return identity()
        }

        if (
            R.empty() || t.empty() ||
            R.rows() != 3 || R.cols() != 3 ||
            t.rows() != 3 || t.cols() != 1
        ) {
            Log.w(
                TAG,
                "recoverPose returned invalid pose: " +
                        "R=${R.rows()}x${R.cols()} empty=${R.empty()}, " +
                        "t=${t.rows()}x${t.cols()} empty=${t.empty()}"
            )
            return identity()
        }

        val finalIndexed = ArrayList<IndexedMatch>()

        val nMask = minOf(originalIndexedMatches.size, poseMask.rows())

        for (i in 0 until nMask) {
            val mv = poseMask.get(i, 0)

            if (mv != null && mv.isNotEmpty() && mv[0] != 0.0) {
                finalIndexed.add(originalIndexedMatches[i])
            }
        }

        if (finalIndexed.size < MIN_INLIERS) {
            Log.w(TAG, "Too few pose-consistent inliers after recoverPose: ${finalIndexed.size}")
            return identity()
        }

        val inlierRatio = finalIndexed.size.toFloat() / originalIndexedMatches.size.toFloat()

        Log.i(
            TAG,
            "Pose estimated — matches: ${originalIndexedMatches.size}, " +
                    "inliers: ${finalIndexed.size}, ratio: ${"%.2f".format(inlierRatio)}"
        )

        matchSet.replaceIndexedMatches(finalIndexed)

        if (inlierRatio < 0.35f) {
            Log.w(
                TAG,
                "Low inlier ratio (${"%.2f".format(inlierRatio)}) " +
                        "— consider checking image overlap or exposure consistency"
            )
        }

        val tx = t.get(0, 0)?.getOrNull(0)
        val ty = t.get(1, 0)?.getOrNull(0)
        val tz = t.get(2, 0)?.getOrNull(0)

        if (tx == null || ty == null || tz == null) {
            Log.w(TAG, "recoverPose returned unreadable translation values")
            return identity()
        }

        val tMag = Math.sqrt(tx.pow(2) + ty.pow(2) + tz.pow(2))

        if (!tMag.isFinite() || tMag < 0.001) {
            Log.w(TAG, "recoverPose returned bad translation magnitude=$tMag → returning identity fallback")
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