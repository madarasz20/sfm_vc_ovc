package com.d2xcp0.sfm_vc_ocv.helper

import android.util.Log
import com.d2xcp0.sfm_vc_ocv.sfm.FeatureObservation
import com.d2xcp0.sfm_vc_ocv.sfm.FeatureTrack
import org.opencv.core.*
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sqrt

class TrackTriangulator(private val K: Mat) {

    companion object {
        private const val TAG = "TrackTriangulator"

        // A legtavolabbi ket nezet sugarai kozti minimalis szog.
        // Kozel parhuzamos sugarakbol a melyseg nagyon pontatlan.
        private const val MIN_ANGLE_DEG = 2.0
    }

    private val fx = K.get(0, 0)[0]
    private val fy = K.get(1, 1)[0]
    private val cx = K.get(0, 2)[0]
    private val cy = K.get(1, 2)[0]

    // Statisztika a logokhoz
    var rejectedAngle = 0
        private set
    var rejectedBehind = 0
        private set

    // N-nezetes linearis DLT: a track OSSZES olyan megfigyeleset hasznalja,
    // amelynek frame-jehez van poz (korabban csak az elso es utolso).
    fun triangulateTrack(
        track: FeatureTrack,
        rotations: List<Mat>,
        translations: List<Mat>
    ): Point3? {
        val obs = track.observations.filter {
            it.frameIndex in rotations.indices &&
                    it.frameIndex in translations.indices &&
                    validPose(rotations[it.frameIndex], translations[it.frameIndex])
        }

        if (obs.size < 2) return null

        return try {
            val p = triangulateDLT(obs, rotations, translations) ?: return null

            if (obs.any { depthInCamera(rotations[it.frameIndex], translations[it.frameIndex], p) <= 0.0 }) {
                rejectedBehind++
                return null
            }

            if (maxRayAngleDeg(p, obs, rotations, translations) < MIN_ANGLE_DEG) {
                rejectedAngle++
                return null
            }

            p
        } catch (e: Exception) {
            Log.w(TAG, "Track ${track.id}: triangulation failed: ${e.message}")
            null
        }
    }

    // Normalizalt kamerakoordinatakban (K^-1 * pixel) a jobb kondicionalasert
    private fun triangulateDLT(
        obs: List<FeatureObservation>,
        rotations: List<Mat>,
        translations: List<Mat>
    ): Point3? {
        val A = Mat(2 * obs.size, 4, CvType.CV_64F)

        obs.forEachIndexed { k, o ->
            val R = rotations[o.frameIndex]
            val t = translations[o.frameIndex]
            val xn = (o.point2D.x - cx) / fx
            val yn = (o.point2D.y - cy) / fy

            for (c in 0 until 4) {
                val p0 = if (c < 3) R.get(0, c)[0] else t.get(0, 0)[0]
                val p1 = if (c < 3) R.get(1, c)[0] else t.get(1, 0)[0]
                val p2 = if (c < 3) R.get(2, c)[0] else t.get(2, 0)[0]
                A.put(2 * k, c, xn * p2 - p0)
                A.put(2 * k + 1, c, yn * p2 - p1)
            }
        }

        val w = Mat()
        val u = Mat()
        val vt = Mat()
        Core.SVDecomp(A, w, u, vt)

        val X = DoubleArray(4) { vt.get(3, it)[0] }

        A.release(); w.release(); u.release(); vt.release()

        if (abs(X[3]) < 1e-12) return null

        val p = Point3(X[0] / X[3], X[1] / X[3], X[2] / X[3])
        return if (p.x.isFinite() && p.y.isFinite() && p.z.isFinite()) p else null
    }

    // A nezetparok kozul a legnagyobb sugarszog (fokban)
    private fun maxRayAngleDeg(
        p: Point3,
        obs: List<FeatureObservation>,
        rotations: List<Mat>,
        translations: List<Mat>
    ): Double {
        val rays = obs.map {
            val C = cameraCenter(rotations[it.frameIndex], translations[it.frameIndex])
            doubleArrayOf(p.x - C[0], p.y - C[1], p.z - C[2])
        }

        var best = 0.0
        for (i in rays.indices) {
            for (j in i + 1 until rays.size) {
                val a = rays[i]
                val b = rays[j]
                val dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
                val n = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) *
                        sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
                if (n < 1e-12) continue
                best = maxOf(best, Math.toDegrees(acos((dot / n).coerceIn(-1.0, 1.0))))
            }
        }
        return best
    }

    // C = -R^T * t
    private fun cameraCenter(R: Mat, t: Mat): DoubleArray = DoubleArray(3) { c ->
        -(R.get(0, c)[0] * t.get(0, 0)[0] +
                R.get(1, c)[0] * t.get(1, 0)[0] +
                R.get(2, c)[0] * t.get(2, 0)[0])
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
