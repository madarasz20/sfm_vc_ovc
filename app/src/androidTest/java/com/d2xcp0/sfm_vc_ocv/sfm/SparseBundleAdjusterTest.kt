package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Point3
import java.util.Random

/**
 * U1, U2: a sajat ritka bundle adjustment ellenorzese szintetikus jeleneten.
 * 6 kamera 10 fokonkent egy koriven, 300 pont, ismert valodi pozok.
 * A kezdoertekeket elrontjuk, es azt vizsgaljuk, hogy a BA visszaall-e a valodi ertekekhez.
 */
@RunWith(AndroidJUnit4::class)
class SparseBundleAdjusterTest {

    companion object {
        private const val TAG = "SFM_UNIT"
        private const val N_POINTS = 300
        private val ANGLES = listOf(0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
    }

    @Before
    fun loadOpenCV() {
        assertTrue("OpenCV not loaded", OpenCVLoader.initLocal())
    }

    private class Setup(
        val trueCams: List<SyntheticScene.Cam>,
        val rotations: Array<Mat?>,
        val translations: Array<Mat?>,
        val points: MutableMap<Int, Point3>,
        val tracks: List<FeatureTrack>
    )

    // Zajos meresek + elrontott kezdoertekek.
    // A 0. kamera a valodi pozban rogzitett; az 1. kamera eltolasa valodi (ez rogziti a skalat),
    // csak a forgatasa romlik. Igy a BA eredmenye kozvetlenul a valodi ertekekkel vetheto ossze.
    private fun setup(noisePx: Double, outlierRatio: Double, rnd: Random): Setup {
        val cams = SyntheticScene.cameras(ANGLES)
        val pts = SyntheticScene.points(N_POINTS, rnd)
        val tracks = SyntheticScene.tracks(cams, pts, noisePx, rnd)

        // Kiugro meresek: a megfigyelesek egy reszet 30 px-lel eltoljuk
        if (outlierRatio > 0) {
            for (track in tracks) {
                for (k in track.observations.indices) {
                    if (rnd.nextDouble() < outlierRatio) {
                        val o = track.observations[k]
                        val a = rnd.nextDouble() * 2 * Math.PI
                        track.observations[k] = o.copy(
                            point2D = Point(o.point2D.x + 30 * Math.cos(a), o.point2D.y + 30 * Math.sin(a))
                        )
                    }
                }
            }
        }

        val rotations = arrayOfNulls<Mat>(cams.size)
        val translations = arrayOfNulls<Mat>(cams.size)
        cams.forEachIndexed { i, cam ->
            val R = if (i == 0) cam.R else SyntheticScene.perturbRotation(cam.R, 0.5, rnd)
            val t = if (i <= 1) cam.t else DoubleArray(3) { cam.t[it] + 0.05 * rnd.nextGaussian() }
            rotations[i] = SyntheticScene.rMat(R)
            translations[i] = SyntheticScene.tMat(t)
        }

        val points = HashMap<Int, Point3>()
        pts.forEachIndexed { j, p ->
            points[j] = Point3(
                p.x + 0.05 * rnd.nextGaussian(),
                p.y + 0.05 * rnd.nextGaussian(),
                p.z + 0.05 * rnd.nextGaussian()
            )
        }
        return Setup(cams, rotations, translations, points, tracks)
    }

    private fun maxRotationErrorDeg(s: Setup): Double =
        s.trueCams.indices.maxOf { i ->
            SyntheticScene.rotationErrorDeg(s.trueCams[i].R, SyntheticScene.rArray(s.rotations[i]!!))
        }

    // Kameraközeppont-hiba a koriv sugaranak (5) aranyaban
    private fun maxCenterErrorRel(s: Setup): Double =
        s.trueCams.indices.maxOf { i ->
            val est = SyntheticScene.Cam(SyntheticScene.rArray(s.rotations[i]!!), SyntheticScene.tArray(s.translations[i]!!))
            SyntheticScene.dist(s.trueCams[i].center(), est.center()) / 5.0
        }

    // U1: konvergencia zajos jeleneten
    @Test
    fun convergesOnNoisyScene() {
        val s = setup(noisePx = 0.5, outlierRatio = 0.0, rnd = Random(42))
        val rotBefore = maxRotationErrorDeg(s)

        val stats = SparseBundleAdjuster(SyntheticScene.K()).adjust(
            s.rotations, s.translations, s.points, s.tracks,
            fixedFrame = 0, scaleFrame = 1, maxIterations = 30
        )
        assertNotNull(stats)
        stats!!

        val rotAfter = maxRotationErrorDeg(s)
        val centerErr = maxCenterErrorRel(s)
        Log.i(
            TAG,
            "U1 BA convergence: rms ${"%.3f".format(stats.initialRms)} -> ${"%.3f".format(stats.finalRms)} px, " +
                    "iters=${stats.iterations}, max rot error ${"%.3f".format(rotBefore)} -> ${"%.4f".format(rotAfter)} deg, " +
                    "max center error ${"%.3f".format(centerErr * 100)} % of radius, ${stats.millis} ms"
        )

        // 0,5 px zajnal a 2D hiba-normak RMS-e kb. 0,5 * sqrt(2) = 0,71 px
        assertTrue("final RMS too high: ${stats.finalRms}", stats.finalRms < 0.8)
        assertTrue("RMS did not decrease", stats.finalRms < stats.initialRms)
        assertTrue("rotation error too high: $rotAfter", rotAfter < 0.1)
        assertTrue("camera center error too high: $centerErr", centerErr < 0.01)
    }

    // U2: kiugro meresek (5%, 30 px) mellett a Huber-veszteseg miatt a pozok pontosak maradnak
    @Test
    fun robustToOutliers() {
        val s = setup(noisePx = 0.5, outlierRatio = 0.05, rnd = Random(7))

        val stats = SparseBundleAdjuster(SyntheticScene.K()).adjust(
            s.rotations, s.translations, s.points, s.tracks,
            fixedFrame = 0, scaleFrame = 1, maxIterations = 30
        )
        assertNotNull(stats)
        stats!!

        val rotAfter = maxRotationErrorDeg(s)
        val centerErr = maxCenterErrorRel(s)
        Log.i(
            TAG,
            "U2 BA outliers (5%, 30 px): rms ${"%.3f".format(stats.initialRms)} -> ${"%.3f".format(stats.finalRms)} px, " +
                    "max rot error ${"%.4f".format(rotAfter)} deg, max center error ${"%.3f".format(centerErr * 100)} % of radius"
        )

        assertTrue("rotation error too high with outliers: $rotAfter", rotAfter < 0.5)
        assertTrue("camera center error too high with outliers: $centerErr", centerErr < 0.02)
    }
}
