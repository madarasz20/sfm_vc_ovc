package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*

class PoseRefiner(private val K: Mat) {

    companion object { private const val TAG = "PoseRefiner" }

    /**
     * SAFE Pose Refinement:
     * - Never breaks the pipeline.
     * - Only refines pose when 3D/2D relations make sense.
     * - Returns original R,t if PnP is weak or inconsistent.
     */
    fun refine(
        pts3d: List<Point3>,
        pts2d: List<Point>,
        R: Mat,
        t: Mat
    ): Triple<List<Point3>, Mat, Mat> {

        //validate
        if (pts3d.size < 6 || pts2d.size < 6) {
            Log.i(TAG, "Not enough points for refinement (${pts3d.size}), skipping.")
            return Triple(pts3d, R, t)
        }

        //match counts
        val n = minOf(pts3d.size, pts2d.size)
        if (n < 6) {
            Log.i(TAG, "Not enough matched pairs ($n), skipping refine.")
            return Triple(pts3d, R, t)
        }

        // Use same number of points from both lists
        val used3D = pts3d.take(n)
        val used2D = pts2d.take(n)

        // Convert to OpenCV structures
        val objPoints = MatOfPoint3f(*used3D.toTypedArray())
        val imgPoints = MatOfPoint2f(*used2D.toTypedArray())

        if (objPoints.rows() < 6 || imgPoints.rows() < 6) {
            Log.i(TAG, "Too few valid rows after conversion, skipping refine.")
            return Triple(pts3d, R, t)
        }

        //setup initial pose
        val rvec = Mat()
        Calib3d.Rodrigues(R, rvec)  //rotation -> rvec

        val tvec = t.clone()

        //Zero distortion
        val dist = MatOfDouble(0.0, 0.0, 0.0, 0.0)

        //try PnP
        val success = try {
            Calib3d.solvePnP(
                objPoints,
                imgPoints,
                K,
                dist,
                rvec,
                tvec,
                true,
                Calib3d.SOLVEPNP_ITERATIVE
            )
        } catch (e: Exception) {
            Log.w(TAG, "solvePnP exception: ${e.message}")
            false
        }

        if (!success) {
            Log.i(TAG, "PnP failed or unstable → keeping original pose.")
            return Triple(pts3d, R, t)
        }

        //Convert refined rvec to Rmat
        val Rref = Mat()
        try {
            Calib3d.Rodrigues(rvec, Rref)
        } catch (e: Exception) {
            Log.w(TAG, "Rodrigues failed: ${e.message}, keeping original R.")
            return Triple(pts3d, R, t)
        }

        Log.i(TAG, "Pose refined successfully using ${objPoints.rows()} correspondences.")

        //return the updated R,t but keep original 3D points for triangulation
        return Triple(pts3d, Rref, tvec)
    }
}
