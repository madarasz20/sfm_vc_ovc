package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*

/**
 * Lightweight Bundle Adjustment using iterative reprojection error minimization.
 *
 * Strategy: Gauss-Newton style iterative refinement.
 * For each iteration:
 *   1. Project all 3D points into all cameras
 *   2. Compute reprojection errors
 *   3. Update camera poses via solvePnPRansac
 *   4. Update 3D points via re-triangulation from best two views
 *
 * This is not full BA (no simultaneous Jacobian), but it converges well
 * for small scenes and runs entirely on-device without external libraries.
 */
class BundleAdjuster(private val K: Mat, private val D: Mat) {

    companion object {
        private const val TAG = "BundleAdjuster"
        private const val MAX_ITERATIONS = 10
        private const val CONVERGENCE_THRESHOLD = 0.001  // stop if improvement < 0.1%
        private const val MAX_REPROJ_ERROR = 8.0         // px — points above this are outliers
        private const val MIN_TRACK_LENGTH = 2           // point must be seen in at least 2 views
    }

    data class BAFrame(
        val index: Int,
        var R: Mat,
        var t: Mat,
        val keypoints: List<Point>,   // 2D observations in this frame
        val pointIndices: List<Int>   // which 3D point each keypoint corresponds to
    )

    data class BAResult(
        val points3D: List<Point3>,
        val rotations: List<Mat>,
        val translations: List<Mat>,
        val finalReprojError: Double
    )

    /**
     * Main entry point.
     *
     * @param points3D      Initial 3D point cloud
     * @param rotations     Per-frame rotation matrices (same order as imgs)
     * @param translations  Per-frame translation vectors
     * @param observations  Map of point_index -> list of (frame_index, 2D point) observations
     */
    fun adjust(
        points3D: MutableList<Point3>,
        rotations: MutableList<Mat>,
        translations: MutableList<Mat>,
        observations: Map<Int, List<Pair<Int, Point>>>
    ): BAResult {

        Log.i(TAG, "Starting BA: ${points3D.size} points, ${rotations.size} frames")

        if (points3D.isEmpty() || rotations.isEmpty()) {
            Log.w(TAG, "BA: empty input, skipping")
            return BAResult(points3D, rotations, translations, Double.MAX_VALUE)
        }

        val distCoeffs = MatOfDouble()
        D.convertTo(distCoeffs, CvType.CV_64F)

        var prevError = computeMeanReprojError(points3D, rotations, translations, observations, distCoeffs)
        Log.i(TAG, "BA initial reprojection error: ${"%.3f".format(prevError)}px")

        for (iter in 0 until MAX_ITERATIONS) {

            // Step 1: Fix 3D points, optimize camera poses
            refineCameraPoses(points3D, rotations, translations, observations, distCoeffs)

            // Step 2: Fix camera poses, optimize 3D points
            refinePoints(points3D, rotations, translations, observations)

            // Step 3: Remove high-error points
            val removed = removeOutlierPoints(points3D, rotations, translations, observations, distCoeffs)

            val currentError = computeMeanReprojError(points3D, rotations, translations, observations, distCoeffs)
            val improvement = (prevError - currentError) / prevError

            Log.i(TAG, "BA iter ${iter+1}: error=${"%.3f".format(currentError)}px " +
                    "improvement=${"%.1f".format(improvement * 100)}% removed=$removed pts")

            if (improvement < CONVERGENCE_THRESHOLD) {
                Log.i(TAG, "BA converged at iteration ${iter+1}")
                break
            }

            prevError = currentError
        }

        val finalError = computeMeanReprojError(points3D, rotations, translations, observations, distCoeffs)
        Log.i(TAG, "BA complete: ${points3D.size} points, final error=${"%.3f".format(finalError)}px")

        return BAResult(points3D, rotations, translations, finalError)
    }

    // ── Step 1: Refine each camera pose given current 3D points ──────────────

    private fun refineCameraPoses(
        points3D: List<Point3>,
        rotations: MutableList<Mat>,
        translations: MutableList<Mat>,
        observations: Map<Int, List<Pair<Int, Point>>>,
        distCoeffs: MatOfDouble
    ) {
        val nFrames = rotations.size

        // Build per-frame observation lists
        // frameObs[f] = list of (Point3, Point2D) visible in frame f
        val frameObs = Array(nFrames) { mutableListOf<Pair<Point3, Point>>() }

        for ((ptIdx, obs) in observations) {
            if (ptIdx >= points3D.size) continue
            val pt3D = points3D[ptIdx]
            for ((frameIdx, pt2D) in obs) {
                if (frameIdx < nFrames) {
                    frameObs[frameIdx].add(pt3D to pt2D)
                }
            }
        }

        // Skip frame 0 — it's the reference frame, keep it fixed
        for (f in 1 until nFrames) {
            val obs = frameObs[f]
            if (obs.size < 6) continue

            val objPts = MatOfPoint3f(*obs.map { it.first }.toTypedArray())
            val imgPts = MatOfPoint2f(*obs.map { it.second }.toTypedArray())

            val rvec = Mat()
            Calib3d.Rodrigues(rotations[f], rvec)
            val tvec = translations[f].clone()

            val inliers = Mat()
            val success = try {
                Calib3d.solvePnPRansac(
                    objPts, imgPts, K, distCoeffs,
                    rvec, tvec,
                    true,   // useExtrinsicGuess
                    50,     // fewer iterations — we're already near the solution
                    MAX_REPROJ_ERROR.toFloat(),
                    0.99,
                    inliers
                )
            } catch (e: Exception) {
                Log.w(TAG, "BA pose refinement failed for frame $f: ${e.message}")
                false
            }

            if (!success || inliers.rows() < 6) continue

            val Rnew = Mat()
            Calib3d.Rodrigues(rvec, Rnew)

            // Sanity check — reject wild pose jumps
            if (poseJumpIsReasonable(rotations[f], Rnew, translations[f], tvec)) {
                rotations[f]    = Rnew
                translations[f] = tvec
            }
        }
    }

    // ── Step 2: Refine each 3D point given current camera poses ──────────────

    private fun refinePoints(
        points3D: MutableList<Point3>,
        rotations: List<Mat>,
        translations: List<Mat>,
        observations: Map<Int, List<Pair<Int, Point>>>
    ) {
        for ((ptIdx, obs) in observations) {
            if (ptIdx >= points3D.size) continue
            if (obs.size < MIN_TRACK_LENGTH) continue

            // Collect all views of this point
            val validObs = obs.filter { (frameIdx, _) -> frameIdx < rotations.size }
            if (validObs.size < 2) continue

            // Re-triangulate from the two views with the largest baseline
            val bestPair = findBestTriangulationPair(validObs, rotations, translations)
            if (bestPair == null) continue

            val (f1, pt1) = bestPair.first
            val (f2, pt2) = bestPair.second

            val refined = triangulatePoint(
                pt1, pt2,
                rotations[f1], translations[f1],
                rotations[f2], translations[f2]
            ) ?: continue

            // Only accept if it reduces reprojection error
            val oldError = reprojErrorForPoint(points3D[ptIdx], obs, rotations, translations)
            val newError = reprojErrorForPoint(refined, obs, rotations, translations)

            if (newError < oldError) {
                points3D[ptIdx] = refined
            }
        }
    }

    // ── Step 3: Remove points with high reprojection error ───────────────────

    private fun removeOutlierPoints(
        points3D: MutableList<Point3>,
        rotations: List<Mat>,
        translations: List<Mat>,
        observations: Map<Int, List<Pair<Int, Point>>>,
        distCoeffs: MatOfDouble
    ): Int {
        val toRemove = mutableSetOf<Int>()

        for ((ptIdx, obs) in observations) {
            if (ptIdx >= points3D.size) continue
            val pt3D = points3D[ptIdx]

            var maxError = 0.0
            for ((frameIdx, pt2D) in obs) {
                if (frameIdx >= rotations.size) continue
                val err = reprojErrorSingle(pt3D, pt2D, rotations[frameIdx], translations[frameIdx])
                if (err > maxError) maxError = err
            }

            if (maxError > MAX_REPROJ_ERROR * 3) {
                toRemove.add(ptIdx)
            }
        }

        // Remove in reverse order to preserve indices
        toRemove.sortedDescending().forEach { idx ->
            if (idx < points3D.size) points3D.removeAt(idx)
        }

        return toRemove.size
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun computeMeanReprojError(
        points3D: List<Point3>,
        rotations: List<Mat>,
        translations: List<Mat>,
        observations: Map<Int, List<Pair<Int, Point>>>,
        distCoeffs: MatOfDouble
    ): Double {
        var totalError = 0.0
        var totalObs   = 0

        for ((ptIdx, obs) in observations) {
            if (ptIdx >= points3D.size) continue
            val pt3D = points3D[ptIdx]

            for ((frameIdx, pt2D) in obs) {
                if (frameIdx >= rotations.size) continue
                val err = reprojErrorSingle(pt3D, pt2D, rotations[frameIdx], translations[frameIdx])
                totalError += err
                totalObs++
            }
        }

        return if (totalObs > 0) totalError / totalObs else Double.MAX_VALUE
    }

    private fun reprojErrorForPoint(
        pt3D: Point3,
        obs: List<Pair<Int, Point>>,
        rotations: List<Mat>,
        translations: List<Mat>
    ): Double {
        var total = 0.0
        var count = 0
        for ((frameIdx, pt2D) in obs) {
            if (frameIdx >= rotations.size) continue
            total += reprojErrorSingle(pt3D, pt2D, rotations[frameIdx], translations[frameIdx])
            count++
        }
        return if (count > 0) total / count else Double.MAX_VALUE
    }

    private fun reprojErrorSingle(
        pt3D: Point3,
        pt2D: Point,
        R: Mat, t: Mat
    ): Double {
        return try {
            val objPts = MatOfPoint3f(pt3D)
            val imgPts = MatOfPoint2f()
            val rvec   = Mat()
            Calib3d.Rodrigues(R, rvec)

            // Zero distortion for speed in inner loop — D already applied during undistortion
            Calib3d.projectPoints(objPts, rvec, t, K, MatOfDouble(0.0,0.0,0.0,0.0,0.0), imgPts)

            val proj = imgPts.toArray()
            if (proj.isEmpty()) Double.MAX_VALUE
            else Math.hypot(proj[0].x - pt2D.x, proj[0].y - pt2D.y)
        } catch (e: Exception) {
            Double.MAX_VALUE
        }
    }

    private fun triangulatePoint(
        pt1: Point, pt2: Point,
        R1: Mat, t1: Mat,
        R2: Mat, t2: Mat
    ): Point3? {
        return try {
            val Rt1 = buildRt(R1, t1)
            val Rt2 = buildRt(R2, t2)
            val P1  = Mat(); Core.gemm(K, Rt1, 1.0, Mat(), 0.0, P1)
            val P2  = Mat(); Core.gemm(K, Rt2, 1.0, Mat(), 0.0, P2)

            val m1  = MatOfPoint2f(pt1)
            val m2  = MatOfPoint2f(pt2)
            val out = Mat()
            Calib3d.triangulatePoints(P1, P2, m1, m2, out)

            val w = out.get(3, 0)[0]
            if (w == 0.0) return null
            val X = out.get(0, 0)[0] / w
            val Y = out.get(1, 0)[0] / w
            val Z = out.get(2, 0)[0] / w

            if (!X.isFinite() || !Y.isFinite() || !Z.isFinite()) null
            else if (Z <= 0) null
            else Point3(X, Y, Z)
        } catch (e: Exception) { null }
    }

    private fun buildRt(R: Mat, t: Mat): Mat {
        val Rt = Mat(3, 4, CvType.CV_64F)
        for (r in 0 until 3) {
            for (c in 0 until 3) Rt.put(r, c, R.get(r, c)[0])
            Rt.put(r, 3, t.get(r, 0)[0])
        }
        return Rt
    }

    private fun findBestTriangulationPair(
        obs: List<Pair<Int, Point>>,
        rotations: List<Mat>,
        translations: List<Mat>
    ): Pair<Pair<Int, Point>, Pair<Int, Point>>? {
        if (obs.size < 2) return null

        var bestAngle = 0.0
        var bestPair: Pair<Pair<Int, Point>, Pair<Int, Point>>? = null

        for (i in obs.indices) {
            for (j in i+1 until obs.size) {
                val (f1, _) = obs[i]
                val (f2, _) = obs[j]
                if (f1 >= rotations.size || f2 >= rotations.size) continue

                // Baseline between camera centres
                val C1 = computeCenter(rotations[f1], translations[f1])
                val C2 = computeCenter(rotations[f2], translations[f2])
                val baseline = Math.sqrt(
                    (C1[0]-C2[0]).let{it*it} +
                            (C1[1]-C2[1]).let{it*it} +
                            (C1[2]-C2[2]).let{it*it}
                )

                if (baseline > bestAngle) {
                    bestAngle = baseline
                    bestPair  = obs[i] to obs[j]
                }
            }
        }
        return bestPair
    }

    private fun computeCenter(R: Mat, t: Mat): DoubleArray {
        val Rt = Mat(); Core.transpose(R, Rt)
        val C  = Mat(); Core.gemm(Rt, t, -1.0, Mat(), 0.0, C)
        return doubleArrayOf(C.get(0,0)[0], C.get(1,0)[0], C.get(2,0)[0])
    }

    // Reject unreasonably large pose jumps during BA
    private fun poseJumpIsReasonable(Rold: Mat, Rnew: Mat, told: Mat, tnew: Mat): Boolean {
        val dt = Mat(); Core.subtract(tnew, told, dt)
        val dMag = Math.sqrt(
            dt.get(0,0)[0].let{it*it} +
                    dt.get(1,0)[0].let{it*it} +
                    dt.get(2,0)[0].let{it*it}
        )
        // Reject if pose jumped more than 30% of original translation magnitude
        val tMag = Math.sqrt(
            told.get(0,0)[0].let{it*it} +
                    told.get(1,0)[0].let{it*it} +
                    told.get(2,0)[0].let{it*it}
        )
        return dMag < tMag * 0.3 + 0.1  // 0.1 = absolute minimum tolerance
    }
}