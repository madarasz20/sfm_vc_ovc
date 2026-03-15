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

        if (R1.empty() || R2.empty() || t1.empty() || t2.empty()) {
            Log.e(TAG, "Empty pose matrix")
            return emptyList()
        }

        if (R1.rows() != 3 || R1.cols() != 3 || R2.rows() != 3 || R2.cols() != 3) {
            Log.e(TAG, "Rotation matrices must be 3x3")
            return emptyList()
        }

        if (t1.rows() != 3 || t1.cols() != 1 || t2.rows() != 3 || t2.cols() != 1) {
            Log.e(TAG, "Translation vectors must be 3x1, got t1=${t1.rows()}x${t1.cols()} t2=${t2.rows()}x${t2.cols()}")
            return emptyList()
        }


        // P1 = K [R1 | t1]
        // P2 = K [R2 | t2]
        // Ensure double precision and correct shapes:
        // R: 3x3, t: 3x1, P: 3x4

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
                val rv1 = getMatValue(R1d, r, c) ?: return emptyList()
                val rv2 = getMatValue(R2d, r, c) ?: return emptyList()
                Rt1.put(r, c, rv1)
                Rt2.put(r, c, rv2)
            }
            val tv1 = getMatValue(t1d, r, 0) ?: return emptyList()
            val tv2 = getMatValue(t2d, r, 0) ?: return emptyList()
            Rt1.put(r, 3, tv1)
            Rt2.put(r, 3, tv2)
        }

        val P1 = Mat()
        val P2 = Mat()
        Core.gemm(K, Rt1, 1.0, Mat(), 0.0, P1)
        Core.gemm(K, Rt2, 1.0, Mat(), 0.0, P2)


        //Convert matched points
        val mat1 = MatOfPoint2f(*pts1.toTypedArray())
        val mat2 = MatOfPoint2f(*pts2.toTypedArray())


        // Triangulate 4D homogeneous output
        // pts4d: 4 x N matrix, each column = [x, y, z, w]^T

        val pts4d = Mat()
        Calib3d.triangulatePoints(P1, P2, mat1, mat2, pts4d)

        if (pts4d.empty() || pts4d.rows() != 4 || pts4d.cols() == 0) {
            Log.e(TAG, "triangulatePoints failed: pts4d size=${pts4d.rows()}x${pts4d.cols()}")
            return emptyList()
        }

        // Convert to euclidean 3D, filter invalid / behind-camera points
        val cloud = mutableListOf<Point3>()
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

            //sanity checks
            if (!X.isFinite() || !Y.isFinite() || !Z.isFinite()) continue

            if (Z <= 0) continue

            //reject crazy far-out points
            if (Z > 5000 || Z < -5000) continue

            cloud.add(Point3(X, Y, Z))
        }

        //feasible depth filtering to remove farout points
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

    private fun getMatValue(mat: Mat, row: Int, col: Int): Double? {
        if (mat.empty()) return null
        if (row < 0 || row >= mat.rows() || col < 0 || col >= mat.cols()) return null

        val v = mat.get(row, col) ?: return null
        if (v.isEmpty()) return null

        return v[0]
    }

}
