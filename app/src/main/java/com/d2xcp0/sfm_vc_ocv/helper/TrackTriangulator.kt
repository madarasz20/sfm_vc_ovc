package com.d2xcp0.sfm_vc_ocv.helper

import android.util.Log
import com.d2xcp0.sfm_vc_ocv.sfm.FeatureTrack
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import kotlin.math.abs

class TrackTriangulator(private val K: Mat) {

    companion object {
        private const val TAG = "TrackTriangulator"
    }

    fun triangulateTrack(
        track: FeatureTrack,
        rotations: List<Mat>,
        translations: List<Mat>
    ): Point3? {
        if (track.observations.size < 2) return null

        val obs = track.observations.sortedBy { it.frameIndex }

        val first = obs.first()
        val last = obs.last()

        if (first.frameIndex !in rotations.indices || last.frameIndex !in rotations.indices) {
            Log.w(TAG, "Track ${track.id}: frame index outside rotations")
            return null
        }

        if (first.frameIndex !in translations.indices || last.frameIndex !in translations.indices) {
            Log.w(TAG, "Track ${track.id}: frame index outside translations")
            return null
        }

        val R1 = rotations[first.frameIndex]
        val t1 = translations[first.frameIndex]
        val R2 = rotations[last.frameIndex]
        val t2 = translations[last.frameIndex]

        if (!validPose(R1, t1) || !validPose(R2, t2)) {
            Log.w(TAG, "Track ${track.id}: invalid pose")
            return null
        }

        val P1 = projectionMatrix(R1, t1)
        val P2 = projectionMatrix(R2, t2)

        val pts1 = MatOfPoint2f(first.point2D)
        val pts2 = MatOfPoint2f(last.point2D)

        val pts4d = Mat()

        return try {
            Calib3d.triangulatePoints(P1, P2, pts1, pts2, pts4d)

            if (pts4d.empty() || pts4d.rows() != 4 || pts4d.cols() < 1) {
                return null
            }

            val w = pts4d.get(3, 0)?.getOrNull(0) ?: return null
            if (abs(w) < 1e-9) return null

            val x = (pts4d.get(0, 0)?.getOrNull(0) ?: return null) / w
            val y = (pts4d.get(1, 0)?.getOrNull(0) ?: return null) / w
            val z = (pts4d.get(2, 0)?.getOrNull(0) ?: return null) / w

            val p = Point3(x, y, z)

            if (!x.isFinite() || !y.isFinite() || !z.isFinite()) {
                return null
            }

            val d1 = depthInCamera(R1, t1, p)
            val d2 = depthInCamera(R2, t2, p)

            if (d1 <= 0.0 || d2 <= 0.0) {
                return null
            }

            p
        } catch (e: Exception) {
            Log.w(TAG, "Track ${track.id}: triangulation failed: ${e.message}")
            null
        } finally {
            P1.release()
            P2.release()
            pts1.release()
            pts2.release()
            pts4d.release()
        }
    }

    private fun projectionMatrix(R: Mat, t: Mat): Mat {
        val R64 = Mat()
        val t64 = Mat()

        R.convertTo(R64, CvType.CV_64F)
        t.convertTo(t64, CvType.CV_64F)

        val Rt = Mat(3, 4, CvType.CV_64F)

        for (r in 0 until 3) {
            for (c in 0 until 3) {
                Rt.put(r, c, R64.get(r, c)[0])
            }
            Rt.put(r, 3, t64.get(r, 0)[0])
        }

        val P = Mat()
        Core.gemm(K, Rt, 1.0, Mat(), 0.0, P)

        R64.release()
        t64.release()
        Rt.release()

        return P
    }

    private fun depthInCamera(R: Mat, t: Mat, p: Point3): Double {
        return R.get(2, 0)[0] * p.x +
                R.get(2, 1)[0] * p.y +
                R.get(2, 2)[0] * p.z +
                t.get(2, 0)[0]
    }

    private fun validPose(R: Mat, t: Mat): Boolean {
        return !R.empty() &&
                !t.empty() &&
                R.rows() == 3 &&
                R.cols() == 3 &&
                t.rows() == 3 &&
                t.cols() == 1
    }
}