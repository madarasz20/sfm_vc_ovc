package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.util.Random
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * U3: a teljes inkrementalis regisztracio (PnP + trianguláció + BA) szintetikus track-eken.
 * 8 kamera 10 fokonkent, 400 pont, 0,3 px zaj. A kezdo par relativ pozat a valodi ertekbol adjuk,
 * a recoverPose-hoz hasonloan egysegnyi hosszu eltolassal.
 *
 * Ellenorzes: minden kep regisztralodik, es a kamerak egymas kozti tavolsag-aranyai egyeznek
 * a valodival – vagyis a skala nem sodrodik (ez volt a regi lancolas hibaja).
 */
@RunWith(AndroidJUnit4::class)
class IncrementalReconstructorTest {

    companion object {
        private const val TAG = "SFM_UNIT"
    }

    @Before
    fun loadOpenCV() {
        assertTrue("OpenCV not loaded", OpenCVLoader.initLocal())
    }

    @Test
    fun registersAllFramesWithConsistentScale() {
        val rnd = Random(123)
        val cams = SyntheticScene.cameras(List(8) { it * 10.0 })
        val pts = SyntheticScene.points(400, rnd)
        val tracks = SyntheticScene.tracks(cams, pts, noisePx = 0.3, rnd = rnd)

        // Kezdo par (0, 1) relativ poza: R01 = R1 * R0^T, t01 = t1 - R01 * t0, egysegnyi hosszra normalva
        val c0 = cams[0]; val c1 = cams[1]
        val r0T = DoubleArray(9) { c0.R[(it % 3) * 3 + it / 3] }
        val r01 = SyntheticScene.mul3(c1.R, r0T)
        val t01 = DoubleArray(3) { r -> c1.t[r] - (r01[3 * r] * c0.t[0] + r01[3 * r + 1] * c0.t[1] + r01[3 * r + 2] * c0.t[2]) }
        val n = sqrt(t01[0] * t01[0] + t01[1] * t01[1] + t01[2] * t01[2])
        for (i in 0 until 3) t01[i] /= n

        val result = IncrementalReconstructor(SyntheticScene.K(), useBundleAdjustment = true).reconstruct(
            numFrames = cams.size,
            tracks = tracks,
            initFrameA = 0,
            initFrameB = 1,
            initR = SyntheticScene.rMat(r01),
            initT = SyntheticScene.tMat(t01)
        )

        assertEquals("not all frames registered", cams.size, result.registeredFrames)

        // Kameraközeppontok tavolsaga a 0. kameratol, az 1. kamera tavolsagahoz viszonyitva
        val estCenters = cams.indices.map { i ->
            SyntheticScene.Cam(
                SyntheticScene.rArray(result.rotations[i]!!),
                SyntheticScene.tArray(result.translations[i]!!)
            ).center()
        }
        val trueCenters = cams.map { it.center() }
        val estBase = SyntheticScene.dist(estCenters[1], estCenters[0])
        val trueBase = SyntheticScene.dist(trueCenters[1], trueCenters[0])

        var maxRatioErr = 0.0
        for (i in 2 until cams.size) {
            val estRatio = SyntheticScene.dist(estCenters[i], estCenters[0]) / estBase
            val trueRatio = SyntheticScene.dist(trueCenters[i], trueCenters[0]) / trueBase
            maxRatioErr = maxOf(maxRatioErr, abs(estRatio - trueRatio) / trueRatio)
        }

        assertNotNull(result.finalBA)
        val rms = result.finalBA!!.finalRms
        Log.i(
            TAG,
            "U3 incremental: registered ${result.registeredFrames}/${cams.size}, points=${result.points.size}, " +
                    "final rms=${"%.3f".format(rms)} px, max distance-ratio error=${"%.3f".format(maxRatioErr * 100)} %"
        )

        assertTrue("scale not consistent: $maxRatioErr", maxRatioErr < 0.01)
        assertTrue("final RMS too high: $rms", rms < 0.6)
    }
}
