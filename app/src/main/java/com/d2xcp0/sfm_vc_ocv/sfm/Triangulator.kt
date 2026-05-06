package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*

class Triangulator(private val K: Mat) {

    companion object {
        private const val TAG = "Triangulator"

        // FIX 1: Tightened from 5000 — for a small object at ~30–50cm,
        // valid Z in normalized camera coords should stay well under 100.
        // Anything beyond this is almost certainly a degenerate triangulation.
        private const val MAX_DEPTH = 100.0

        // FIX 2: Minimum triangulation angle in degrees.
        // Points triangulated from nearly parallel rays are numerically
        // unstable — they pass the Z > 0 check but land at wrong depths.
        private const val MIN_ANGLE_DEG = 1.0
    }

    fun triangulate(
        matches: MatchSet,
        R1: Mat, t1: Mat,
        R2: Mat, t2: Mat
    ): List<Point3> {

        val (pts1, pts2) = matches.getMatchedPoints()

        if (pts1.isEmpty() || pts2.isEmpty() || pts1.size != pts2.size) {
            Log.w(TAG, "No matched points or size mismatch for triangulation.")
            return emptyList()
        }

        if (R1.empty() || R2.empty() || t1.empty() || t2.empty()) {
            Log.e(TAG, "Empty pose matrix")
            return emptyList()
        }

        if (R1.rows() != 3 || R1.cols() != 3 || R2.rows() != 3 || R2.cols() != 3) {
            Log.e(TAG, "Rotation matrices must be 3x3")
            return emptyList()
        }

        if (t1.rows() != 3 || t1.cols() != 1 || t2.rows() != 3 || t2.cols() != 1) {
            Log.e(TAG, "Translation vectors must be 3x1, " +
                    "got t1=${t1.rows()}x${t1.cols()} t2=${t2.rows()}x${t2.cols()}")
            return emptyList()
        }

        val R1d = Mat(); val R2d = Mat()
        val t1d = Mat(); val t2d = Mat()
        R1.convertTo(R1d, CvType.CV_64F)
        R2.convertTo(R2d, CvType.CV_64F)
        t1.convertTo(t1d, CvType.CV_64F)
        t2.convertTo(t2d, CvType.CV_64F)

        val Rt1 = Mat(3, 4, CvType.CV_64F)
        val Rt2 = Mat(3, 4, CvType.CV_64F)

        for (r in 0 until 3) {
            for (c in 0 until 3) {
                Rt1.put(r, c, getMatValue(R1d, r, c) ?: return emptyList())
                Rt2.put(r, c, getMatValue(R2d, r, c) ?: return emptyList())
            }
            Rt1.put(r, 3, getMatValue(t1d, r, 0) ?: return emptyList())
            Rt2.put(r, 3, getMatValue(t2d, r, 0) ?: return emptyList())
        }

        val P1 = Mat(); val P2 = Mat()
        Core.gemm(K, Rt1, 1.0, Mat(), 0.0, P1)
        Core.gemm(K, Rt2, 1.0, Mat(), 0.0, P2)

        val mat1 = MatOfPoint2f(*pts1.toTypedArray())
        val mat2 = MatOfPoint2f(*pts2.toTypedArray())

        val pts4d = Mat()
        Calib3d.triangulatePoints(P1, P2, mat1, mat2, pts4d)

        if (pts4d.empty() || pts4d.rows() != 4 || pts4d.cols() == 0) {
            Log.e(TAG, "triangulatePoints failed: ${pts4d.rows()}x${pts4d.cols()}")
            return emptyList()
        }

        // Extract camera centres for angle computation
        // Cam1 centre = -R1^T * t1, Cam2 centre = -R2^T * t2
        val C1 = computeCameraCenter(R1d, t1d)
        val C2 = computeCameraCenter(R2d, t2d)

        val cloud = mutableListOf<Point3>()
        var rejectedBehind  = 0
        var rejectedDepth   = 0
        var rejectedAngle   = 0
        var rejectedInvalid = 0

        val n = pts4d.cols()
        for (i in 0 until n) {
            val x = getMatValue(pts4d, 0, i) ?: continue
            val y = getMatValue(pts4d, 1, i) ?: continue
            val z = getMatValue(pts4d, 2, i) ?: continue
            val w = getMatValue(pts4d, 3, i) ?: continue

            if (w == 0.0) continue

            val X = x / w
            val Y = y / w
            val Z = z / w

            if (!X.isFinite() || !Y.isFinite() || !Z.isFinite()) {
                rejectedInvalid++
                continue
            }

            // Must be in front of camera 1
            if (Z <= 0) {
                rejectedBehind++
                continue
            }

            // FIX 1: Tighter depth bound appropriate for small object scanning
            if (Z > MAX_DEPTH) {
                rejectedDepth++
                continue
            }

            // FIX 2: Triangulation angle check.
            // Points from nearly parallel rays are numerically unreliable.
            // Even if Z > 0, they can be at wildly wrong depths.
            val angle = triangulationAngle(X, Y, Z, C1, C2)
            if (angle < MIN_ANGLE_DEG) {
                rejectedAngle++
                continue
            }

            cloud.add(Point3(X, Y, Z))
        }

        Log.i(TAG, "Raw triangulated: ${cloud.size}/$n " +
                "(behind=$rejectedBehind, depth=$rejectedDepth, " +
                "angle=$rejectedAngle, invalid=$rejectedInvalid)")

        // FIX 3: MAD-based outlier filter on Z depth (your already-applied fix,
        // kept here). This is correct — using depth not distance from origin.
        val filtered = filterByMAD(cloud)
        Log.i(TAG, "After MAD filter: ${filtered.size}/${cloud.size}")

        return filtered
    }

    // Compute camera centre in world coordinates: C = -R^T * t
    private fun computeCameraCenter(R: Mat, t: Mat): DoubleArray {
        val Rt = Mat()
        Core.transpose(R, Rt)
        val C = Mat()
        Core.gemm(Rt, t, -1.0, Mat(), 0.0, C)
        return doubleArrayOf(C.get(0,0)[0], C.get(1,0)[0], C.get(2,0)[0])
    }

    // Angle (degrees) between the two rays from each camera centre to point P
    private fun triangulationAngle(
        X: Double, Y: Double, Z: Double,
        C1: DoubleArray, C2: DoubleArray
    ): Double {
        // Ray from C1 to P
        val r1 = doubleArrayOf(X - C1[0], Y - C1[1], Z - C1[2])
        // Ray from C2 to P
        val r2 = doubleArrayOf(X - C2[0], Y - C2[1], Z - C2[2])

        val dot = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]
        val mag1 = Math.sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2])
        val mag2 = Math.sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2])

        if (mag1 < 1e-9 || mag2 < 1e-9) return 0.0

        val cosAngle = (dot / (mag1 * mag2)).coerceIn(-1.0, 1.0)
        return Math.toDegrees(Math.acos(cosAngle))
    }

    // MAD-based depth filter — robust to outliers unlike percentile clipping
    private fun filterByMAD(points: List<Point3>): List<Point3> {
        if (points.size < 10) return points

        val depths = points.map { it.z }.sorted()
        val n = depths.size
        val median = depths[n / 2]
        val mad = depths.map { Math.abs(it - median) }.sorted()[n / 2]

        // Degenerate case: all points at same depth (e.g. flat wall)
        // Don't filter in this case — keep everything
        if (mad < 1e-9) return points

        val threshold = 3.0 * (mad / 0.6745)

        return points.filter {
            Math.abs(it.z - median) < threshold && it.z > 0
        }
    }

    private fun getMatValue(mat: Mat, row: Int, col: Int): Double? {
        if (mat.empty()) return null
        if (row < 0 || row >= mat.rows() || col < 0 || col >= mat.cols()) return null
        val v = mat.get(row, col) ?: return null
        if (v.isEmpty()) return null
        return v[0]
    }
}