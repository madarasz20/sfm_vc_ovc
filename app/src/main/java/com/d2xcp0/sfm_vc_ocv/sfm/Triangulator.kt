package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*

class Triangulator(private val K: Mat) {

    companion object {
        private const val TAG = "Triangulator"
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

        // ---------------------------------------------------------
        // P1 = K [R1 | t1]
        // P2 = K [R2 | t2]
        // Ensure double precision and correct shapes:
        // R: 3x3, t: 3x1, P: 3x4
        // ---------------------------------------------------------
        val R1d = Mat()
        val R2d = Mat()
        val t1d = Mat()
        val t2d = Mat()
        R1.convertTo(R1d, CvType.CV_64F)
        R2.convertTo(R2d, CvType.CV_64F)
        t1.convertTo(t1d, CvType.CV_64F)
        t2.convertTo(t2d, CvType.CV_64F)

        val Rt1 = Mat(3, 4, CvType.CV_64F)
        val Rt2 = Mat(3, 4, CvType.CV_64F)

        // [R | t] for camera 1
        for (r in 0 until 3) {
            for (c in 0 until 3) {
                Rt1.put(r, c, R1d.get(r, c)[0])
                Rt2.put(r, c, R2d.get(r, c)[0])
            }
            Rt1.put(r, 3, t1d.get(r, 0)[0])
            Rt2.put(r, 3, t2d.get(r, 0)[0])
        }

        val P1 = Mat()
        val P2 = Mat()
        Core.gemm(K, Rt1, 1.0, Mat(), 0.0, P1)
        Core.gemm(K, Rt2, 1.0, Mat(), 0.0, P2)

        // ---------------------------------------------------------
        // Convert matched points to MatOfPoint2f
        // ---------------------------------------------------------
        val mat1 = MatOfPoint2f(*pts1.toTypedArray())
        val mat2 = MatOfPoint2f(*pts2.toTypedArray())

        // ---------------------------------------------------------
        // Triangulate 4D homogeneous output
        // pts4d: 4 x N matrix, each column = [x, y, z, w]^T
        // ---------------------------------------------------------
        val pts4d = Mat()
        Calib3d.triangulatePoints(P1, P2, mat1, mat2, pts4d)

        // ---------------------------------------------------------
        // Convert to Euclidean 3D, filter invalid / behind-camera points
        // ---------------------------------------------------------
        val cloud = mutableListOf<Point3>()
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

            // basic sanity checks
            if (!X.isFinite() || !Y.isFinite() || !Z.isFinite()) continue

            // we want points in front of the first camera
            if (Z <= 0) continue

            // reject crazy far-out points (scale from homography is arbitrary)
            if (Z > 5000 || Z < -5000) continue

            cloud.add(Point3(X, Y, Z))
        }

        // --- Feasible depth filtering (remove extreme near/far points) ---
        val filtered = filterFeasibleDepth(cloud)
        Log.i(TAG, "Feasible depth points: ${filtered.size}/${cloud.size}")

        return filtered
    }

    private fun normalizePointCloud(points: List<Point3>): List<Point3> {
        if (points.isEmpty()) return points

        var cx = 0.0
        var cy = 0.0
        var cz = 0.0
        for (p in points) {
            cx += p.x
            cy += p.y
            cz += p.z
        }
        cx /= points.size
        cy /= points.size
        cz /= points.size

        var maxDist = 0.0
        val centered = points.map { p ->
            val x = p.x - cx
            val y = p.y - cy
            val z = p.z - cz
            val d = Math.sqrt(x * x + y * y + z * z)
            if (d > maxDist) maxDist = d
            Point3(x, y, z)
        }

        if (maxDist == 0.0) return centered

        val scale = 1.0 / maxDist
        return centered.map { p ->
            Point3(p.x * scale, p.y * scale, p.z * scale)
        }
    }
    private fun filterFeasibleDepth(points: List<Point3>): List<Point3> {
        if (points.size < 10) return points

        val dists = points.map {
            Math.sqrt(it.x*it.x + it.y*it.y + it.z*it.z)
        }.sorted()

        val n = dists.size
        val dMin = dists[(0.1 * (n - 1)).toInt()]
        val dMax = dists[(0.9 * (n - 1)).toInt()]

        return points.filter {
            val d = Math.sqrt(it.x*it.x + it.y*it.y + it.z*it.z)
            d in dMin..dMax
        }
    }

}
