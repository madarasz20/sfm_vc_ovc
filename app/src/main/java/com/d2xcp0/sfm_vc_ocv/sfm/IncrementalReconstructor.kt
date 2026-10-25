package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Inkrementalis SfM: a kamerapozok PnP-vel regisztralodnak a mar meglevo 3D track pontokhoz.
 *
 * Miert kell: a recoverPose minden parnal |t| = 1 egysegvektort ad, igy a parok lancolasakor
 * (tglobal = Rrel * tprev + trel) minden par mas skalaban van -> a targy tobbszor jelenik meg.
 * Itt a skalat csak az init par rogziti; minden tovabbi kamera a meglevo 3D pontokhoz
 * igazodik (solvePnPRansac), igy a skala vegig konzisztens.
 *
 * Feltetelezi, hogy a track 2D pontjai torzitasmentesek (pinhole, D = 0).
 */
class IncrementalReconstructor(private val K: Mat) {

    companion object {
        private const val TAG = "SFM_INCREMENTAL"
        private const val MIN_PNP_POINTS = 12
        private const val MIN_PNP_INLIERS = 10
        private const val PNP_REPROJ_PX = 3.0f
        private const val PNP_ITERATIONS = 500
        private const val MAX_REPROJ_PX = 2.5
        private const val MIN_ANGLE_DEG = 2.0
    }

    data class Result(
        val points: List<Point3>,
        val pointTracks: List<FeatureTrack>,
        val rotations: List<Mat?>,      // null = a frame nem regisztralodott
        val translations: List<Mat?>
    ) {
        val registeredFrames: Int get() = rotations.count { it != null }
    }

    private val fx = K.get(0, 0)[0]
    private val fy = K.get(1, 1)[0]
    private val cx = K.get(0, 2)[0]
    private val cy = K.get(1, 2)[0]

    fun reconstruct(
        numFrames: Int,
        tracks: List<FeatureTrack>,
        initFrameA: Int,
        initFrameB: Int,
        initR: Mat,
        initT: Mat
    ): Result {
        val rotations = arrayOfNulls<Mat>(numFrames)
        val translations = arrayOfNulls<Mat>(numFrames)

        rotations[initFrameA] = Mat.eye(3, 3, CvType.CV_64F)
        translations[initFrameA] = Mat.zeros(3, 1, CvType.CV_64F)
        rotations[initFrameB] = to64(initR)
        translations[initFrameB] = to64(initT)

        // trackId -> 3D pont
        val points = HashMap<Int, Point3>()
        triangulateNewTracks(tracks, rotations, translations, points)
        Log.i(TAG, "init pair $initFrameA-$initFrameB: ${points.size} points")

        // Regisztralas az init partol elore, majd visszafele
        val order = (initFrameB + 1 until numFrames) + (initFrameA - 1 downTo 0)

        for (f in order) {
            val neighbour = if (f > initFrameB) f - 1 else f + 1
            val ok = registerFrame(f, neighbour, tracks, points, rotations, translations)
            if (!ok) {
                Log.w(TAG, "frame $f: registration failed, skipped")
                continue
            }

            val before = points.size
            triangulateNewTracks(tracks, rotations, translations, points)
            val removed = filterPoints(tracks, rotations, translations, points)
            Log.i(
                TAG,
                "frame $f registered: +${points.size - before + removed} new, -$removed bad, total=${points.size}"
            )
        }

        val byId = tracks.associateBy { it.id }
        val ids = points.keys.sorted()
        val result = Result(
            points = ids.map { points.getValue(it) },
            pointTracks = ids.map { byId.getValue(it) },
            rotations = rotations.toList(),
            translations = translations.toList()
        )
        Log.i(TAG, "done: registered ${result.registeredFrames}/$numFrames frames, ${result.points.size} points")
        return result
    }

    private fun registerFrame(
        f: Int,
        neighbour: Int,
        tracks: List<FeatureTrack>,
        points: Map<Int, Point3>,
        rotations: Array<Mat?>,
        translations: Array<Mat?>
    ): Boolean {
        val obj = ArrayList<Point3>()
        val img = ArrayList<Point>()

        for (track in tracks) {
            val p = points[track.id] ?: continue
            val obs = track.observations.firstOrNull { it.frameIndex == f } ?: continue
            obj.add(p)
            img.add(obs.point2D)
        }

        if (obj.size < MIN_PNP_POINTS) {
            Log.w(TAG, "frame $f: only ${obj.size} 2D-3D correspondences")
            return false
        }

        val objMat = MatOfPoint3f(*obj.toTypedArray())
        val imgMat = MatOfPoint2f(*img.toTypedArray())
        val noDist = MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0)

        // Kezdeti becsles: a szomszed (mar regisztralt) frame pozja
        val rvec = Mat()
        val tvec = Mat()
        val Rn = rotations[neighbour]
        val tn = translations[neighbour]
        val useGuess = Rn != null && tn != null
        if (Rn != null && tn != null) {
            Calib3d.Rodrigues(Rn, rvec)
            tn.copyTo(tvec)
        }

        val inliers = Mat()
        val success = Calib3d.solvePnPRansac(
            objMat, imgMat, K, noDist, rvec, tvec,
            useGuess, PNP_ITERATIONS, PNP_REPROJ_PX, 0.999, inliers,
            Calib3d.SOLVEPNP_ITERATIVE
        )

        if (!success || inliers.rows() < MIN_PNP_INLIERS) {
            Log.w(TAG, "frame $f: PnP failed (success=$success, inliers=${inliers.rows()}/${obj.size})")
            return false
        }

        // LM finomitas csak az inliereken
        val inObj = ArrayList<Point3>()
        val inImg = ArrayList<Point>()
        for (r in 0 until inliers.rows()) {
            val idx = inliers.get(r, 0)[0].toInt()
            inObj.add(obj[idx])
            inImg.add(img[idx])
        }
        Calib3d.solvePnPRefineLM(
            MatOfPoint3f(*inObj.toTypedArray()),
            MatOfPoint2f(*inImg.toTypedArray()),
            K, noDist, rvec, tvec
        )

        val R = Mat()
        Calib3d.Rodrigues(rvec, R)
        rotations[f] = to64(R)
        translations[f] = to64(tvec)

        Log.i(TAG, "frame $f: PnP inliers ${inliers.rows()}/${obj.size}")
        return true
    }

    // Uj pontok: minden track, aminek meg nincs 3D pontja, de >= 2 regisztralt kepen latszik
    private fun triangulateNewTracks(
        tracks: List<FeatureTrack>,
        rotations: Array<Mat?>,
        translations: Array<Mat?>,
        points: MutableMap<Int, Point3>
    ) {
        for (track in tracks) {
            if (points.containsKey(track.id)) continue

            val obs = track.observations.filter { rotations.getOrNull(it.frameIndex) != null }
            if (obs.size < 2) continue

            val p = triangulateDLT(obs, rotations, translations) ?: continue

            if (obs.any { depth(p, rotations[it.frameIndex]!!, translations[it.frameIndex]!!) <= 0.0 }) continue
            if (maxRayAngleDeg(p, obs, rotations, translations) < MIN_ANGLE_DEG) continue
            if (obs.any { reprojError(p, it, rotations, translations) > MAX_REPROJ_PX }) continue

            points[track.id] = p
        }
    }

    // Uj kamera utan a mar meglevo pontokat is ujraellenorizzuk; visszaadja a torolt pontok szamat
    private fun filterPoints(
        tracks: List<FeatureTrack>,
        rotations: Array<Mat?>,
        translations: Array<Mat?>,
        points: MutableMap<Int, Point3>
    ): Int {
        val byId = tracks.associateBy { it.id }
        val bad = points.filter { (id, p) ->
            val obs = byId.getValue(id).observations.filter { rotations.getOrNull(it.frameIndex) != null }
            obs.map { reprojError(p, it, rotations, translations) }.average() > MAX_REPROJ_PX * 1.5
        }.keys
        bad.forEach { points.remove(it) }
        return bad.size
    }

    // N-nezetes linearis DLT, normalizalt kamerakoordinatakban
    private fun triangulateDLT(
        obs: List<FeatureObservation>,
        rotations: Array<Mat?>,
        translations: Array<Mat?>
    ): Point3? {
        val A = Mat(2 * obs.size, 4, CvType.CV_64F)

        obs.forEachIndexed { k, o ->
            val R = rotations[o.frameIndex]!!
            val t = translations[o.frameIndex]!!
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

        val w = Mat(); val u = Mat(); val vt = Mat()
        Core.SVDecomp(A, w, u, vt)
        val X = DoubleArray(4) { vt.get(3, it)[0] }
        A.release(); w.release(); u.release(); vt.release()

        if (abs(X[3]) < 1e-12) return null

        val p = Point3(X[0] / X[3], X[1] / X[3], X[2] / X[3])
        return if (p.x.isFinite() && p.y.isFinite() && p.z.isFinite()) p else null
    }

    private fun reprojError(
        p: Point3,
        o: FeatureObservation,
        rotations: Array<Mat?>,
        translations: Array<Mat?>
    ): Double {
        val c = toCamera(p, rotations[o.frameIndex]!!, translations[o.frameIndex]!!)
        if (c[2] <= 0.0) return Double.MAX_VALUE
        val u = fx * c[0] / c[2] + cx
        val v = fy * c[1] / c[2] + cy
        return hypot(u - o.point2D.x, v - o.point2D.y)
    }

    private fun maxRayAngleDeg(
        p: Point3,
        obs: List<FeatureObservation>,
        rotations: Array<Mat?>,
        translations: Array<Mat?>
    ): Double {
        val rays = obs.map {
            val C = center(rotations[it.frameIndex]!!, translations[it.frameIndex]!!)
            doubleArrayOf(p.x - C[0], p.y - C[1], p.z - C[2])
        }
        var best = 0.0
        for (i in rays.indices) for (j in i + 1 until rays.size) {
            val a = rays[i]; val b = rays[j]
            val dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
            val n = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) *
                    sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
            if (n < 1e-12) continue
            best = maxOf(best, Math.toDegrees(acos((dot / n).coerceIn(-1.0, 1.0))))
        }
        return best
    }

    private fun toCamera(p: Point3, R: Mat, t: Mat): DoubleArray = DoubleArray(3) { r ->
        R.get(r, 0)[0] * p.x + R.get(r, 1)[0] * p.y + R.get(r, 2)[0] * p.z + t.get(r, 0)[0]
    }

    private fun depth(p: Point3, R: Mat, t: Mat): Double = toCamera(p, R, t)[2]

    // C = -R^T * t
    private fun center(R: Mat, t: Mat): DoubleArray = DoubleArray(3) { c ->
        -(R.get(0, c)[0] * t.get(0, 0)[0] + R.get(1, c)[0] * t.get(1, 0)[0] + R.get(2, c)[0] * t.get(2, 0)[0])
    }

    private fun to64(m: Mat): Mat {
        val out = Mat()
        m.convertTo(out, CvType.CV_64F)
        return out
    }
}
