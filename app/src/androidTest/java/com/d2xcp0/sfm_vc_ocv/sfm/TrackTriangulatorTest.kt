package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.d2xcp0.sfm_vc_ocv.helper.TrackTriangulator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Point3
import java.util.Random

/**
 * U6, U7: a tobbnezetes linearis (DLT) trianguláció es a latoszog-szures.
 */
@RunWith(AndroidJUnit4::class)
class TrackTriangulatorTest {

    companion object {
        private const val TAG = "SFM_UNIT"
    }

    @Before
    fun loadOpenCV() {
        assertTrue("OpenCV not loaded", OpenCVLoader.initLocal())
    }

    // U6: 3 kamera, zajmentes vetitesek -> a pontot pontosan vissza kell kapni
    @Test
    fun recoversPointFromThreeViews() {
        val cams = SyntheticScene.cameras(listOf(0.0, 15.0, 30.0))
        val p = Point3(0.3, -0.2, 0.1)
        val track = SyntheticScene.tracks(cams, listOf(p), noisePx = 0.0, rnd = Random(1))[0]

        val est = TrackTriangulator(SyntheticScene.K()).triangulateTrack(
            track,
            cams.map { SyntheticScene.rMat(it.R) },
            cams.map { SyntheticScene.tMat(it.t) }
        )

        assertNotNull(est)
        val err = SyntheticScene.dist(
            doubleArrayOf(p.x, p.y, p.z),
            doubleArrayOf(est!!.x, est.y, est.z)
        )
        Log.i(TAG, "U6 DLT triangulation (3 views, no noise): error=${"%.2e".format(err)}")
        assertTrue("triangulation error too high: $err", err < 1e-6)
    }

    // U7: ket majdnem azonos helyu kamera (0,3 fok kulonbseg) -> 2 fok alatti latoszog, el kell utasitani
    @Test
    fun rejectsSmallTriangulationAngle() {
        val cams = SyntheticScene.cameras(listOf(0.0, 0.3))
        val track = SyntheticScene.tracks(cams, listOf(Point3(0.1, 0.1, 0.0)), noisePx = 0.0, rnd = Random(1))[0]

        val triangulator = TrackTriangulator(SyntheticScene.K())
        val est = triangulator.triangulateTrack(
            track,
            cams.map { SyntheticScene.rMat(it.R) },
            cams.map { SyntheticScene.tMat(it.t) }
        )

        Log.i(TAG, "U7 small angle: result=${est}, rejectedAngle=${triangulator.rejectedAngle}")
        assertNull(est)
        assertEquals(1, triangulator.rejectedAngle)
    }
}
