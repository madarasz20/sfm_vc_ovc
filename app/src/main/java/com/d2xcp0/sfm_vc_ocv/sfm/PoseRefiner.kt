package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*

// FIX 1: Accept D (distortion matrix) as a parameter.
// Previously hardcoded to zero distortion, ignoring the calibration
// you worked hard to compute. Wrong distortion = wrong refined poses.
class PoseRefiner(private val K: Mat, private val D: Mat) {

    companion object {
        private const val TAG = "PoseRefiner"
        private const val MIN_POINTS = 8
        private const val MIN_INLIER_RATIO = 0.4f
        private const val MAX_REPROJ_ERROR_PX = 8.0
        private const val RANSAC_REPROJ_THRESHOLD = 4.0f
        private const val RANSAC_ITERATIONS = 100
        private const val RANSAC_CONFIDENCE = 0.99
    }

    fun refine(
        pts3d: List<Point3>,
        pts2d: List<Point>,
        R: Mat,
        t: Mat
    ): Triple<List<Point3>, Mat, Mat> {

        val fallback = Triple(pts3d, R, t)

        // Convert D from Mat to MatOfDouble as required by solvePnPRansac/projectPoints
        val distCoeffs = MatOfDouble()
        D.convertTo(distCoeffs, CvType.CV_64F)

        if (pts3d.size < MIN_POINTS || pts2d.size < MIN_POINTS) {
            Log.i(TAG, "Not enough points for refinement (${pts3d.size}), skipping.")
            return fallback
        }

        val n = minOf(pts3d.size, pts2d.size)
        val used3D = pts3d.take(n)
        val used2D = pts2d.take(n)

        val objPoints = MatOfPoint3f(*used3D.toTypedArray())
        val imgPoints = MatOfPoint2f(*used2D.toTypedArray())

        if (objPoints.rows() < MIN_POINTS || imgPoints.rows() < MIN_POINTS) {
            Log.i(TAG, "Too few valid rows after conversion, skipping refine.")
            return fallback
        }

        val rvec = Mat()
        Calib3d.Rodrigues(R, rvec)
        val tvec = t.clone()

        // FIX 2: Use solvePnPRansac instead of solvePnP.
        // SOLVEPNP_ITERATIVE trusts ALL points equally — including noisy
        // triangulated points. RANSAC rejects outliers before solving,
        // producing a much more stable pose estimate.
        val inliersMat = Mat()
        val success = try {
            Calib3d.solvePnPRansac(
                objPoints,
                imgPoints,
                K,
                distCoeffs,                          // FIX 1: real distortion coefficients
                rvec,
                tvec,
                true,                       // useExtrinsicGuess — seed with current pose
                RANSAC_ITERATIONS,
                RANSAC_REPROJ_THRESHOLD,    // 4px reprojection threshold
                RANSAC_CONFIDENCE,
                inliersMat
            )
        } catch (e: Exception) {
            Log.w(TAG, "solvePnPRansac exception: ${e.message}")
            return fallback
        }

        if (!success) {
            Log.i(TAG, "PnPRansac failed → keeping original pose.")
            return fallback
        }


        // FIX 3: Check inlier ratio — low ratio means the 3D points
        // are too noisy for a reliable pose estimate.
        val inlierCount = inliersMat.rows()
        val inlierRatio = inlierCount.toFloat() / n

        Log.i(TAG, "PnP inliers: $inlierCount/$n (ratio=${"%.2f".format(inlierRatio)})")

        if (inlierRatio < MIN_INLIER_RATIO) {
            Log.w(TAG, "PnP inlier ratio too low (${"%.2f".format(inlierRatio)}) → keeping original pose")
            return fallback
        }

        // Convert refined rvec back to rotation matrix
        val Rref = Mat()
        try {
            Calib3d.Rodrigues(rvec, Rref)
        } catch (e: Exception) {
            Log.w(TAG, "Rodrigues conversion failed: ${e.message} → keeping original pose")
            return fallback
        }

        // FIX 4: Verify reprojection error after refinement.
        // solvePnP success only means convergence — not that it converged
        // to the right answer. High reprojection error = wrong pose.
        val avgReproj = computeReprojectionError(objPoints, imgPoints, rvec, tvec, distCoeffs)
        Log.i(TAG, "Avg reprojection error: ${"%.2f".format(avgReproj)}px")

        if (avgReproj > MAX_REPROJ_ERROR_PX) {
            Log.w(TAG, "Reprojection error too high (${avgReproj}px) → keeping original pose")
            return fallback
        }

        Log.i(TAG, "Pose refined: $inlierCount inliers, " +
                "${"%.2f".format(avgReproj)}px reprojection error")

        return Triple(pts3d, Rref, tvec)
    }

    // Compute mean reprojection error over all points using the refined pose
    private fun computeReprojectionError(
        objPoints: MatOfPoint3f,
        imgPoints: MatOfPoint2f,
        rvec: Mat,
        tvec: Mat,
        distCoeffs: MatOfDouble
    ): Double {
        return try {
            val projected = MatOfPoint2f()
            Calib3d.projectPoints(objPoints, rvec, tvec, K, distCoeffs, projected)

            val observed = imgPoints.toArray()
            val projArr  = projected.toArray()

            if (observed.isEmpty() || projArr.size != observed.size) return Double.MAX_VALUE

            observed.zip(projArr).map { (obs, proj) ->
                Math.hypot(obs.x - proj.x, obs.y - proj.y)
            }.average()
        } catch (e: Exception) {
            Log.w(TAG, "Reprojection error computation failed: ${e.message}")
            Double.MAX_VALUE
        }
    }
}