package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.d2xcp0.sfm_vc_ocv.helper.KeypointUndistortion
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point3
import kotlin.math.hypot

/**
 * U8: a kulcspontok torzitasmentesitese.
 * Ismert, torzitas nelkuli kepponthalmazt az OpenCV torzitasi modelljevel eltorzitunk
 * (projectPoints), majd a rendszer fuggvenyevel torzitasmentesitjuk. Az eredeti pontokat
 * kell visszakapni, a kulcspontok sorrendjenek (es igy a leiro-indexeknek) valtozatlanul.
 */
@RunWith(AndroidJUnit4::class)
class KeypointUndistortionTest {

    companion object {
        private const val TAG = "SFM_UNIT"
    }

    @Before
    fun loadOpenCV() {
        assertTrue("OpenCV not loaded", OpenCVLoader.initLocal())
    }

    @Test
    fun undistortionInvertsDistortion() {
        val K = SyntheticScene.K()
        // Mersekelt, telefonkamerahoz hasonlo torzitas (OpenCV sorrend: k1, k2, p1, p2, k3)
        val D = MatOfDouble(-0.05, 0.01, 0.0005, -0.0005, 0.0)

        // Racs a kep belsejeben (normalizalt koordinatak, z = 1)
        val ideal3D = ArrayList<Point3>()
        for (gx in -5..5) for (gy in -4..4) ideal3D.add(Point3(gx * 0.06, gy * 0.06, 1.0))

        val rvec = Mat.zeros(3, 1, CvType.CV_64F)
        val tvec = Mat.zeros(3, 1, CvType.CV_64F)

        val distorted = MatOfPoint2f()
        Calib3d.projectPoints(MatOfPoint3f(*ideal3D.toTypedArray()), rvec, tvec, K, D, distorted)
        val ideal = MatOfPoint2f()
        Calib3d.projectPoints(MatOfPoint3f(*ideal3D.toTypedArray()), rvec, tvec, K, MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0), ideal)

        val kps = MatOfKeyPoint(*distorted.toArray().map { KeyPoint(it.x.toFloat(), it.y.toFloat(), 1f) }.toTypedArray())
        val result = KeypointUndistortion.undistortKeypoints(kps, K, D).toArray()
        val expected = ideal.toArray()

        assertEquals(expected.size, result.size)

        var maxErr = 0.0
        var maxShift = 0.0
        val dist = distorted.toArray()
        for (i in expected.indices) {
            maxErr = maxOf(maxErr, hypot(result[i].pt.x - expected[i].x, result[i].pt.y - expected[i].y))
            maxShift = maxOf(maxShift, hypot(dist[i].x - expected[i].x, dist[i].y - expected[i].y))
        }
        Log.i(
            TAG,
            "U8 undistortion: ${expected.size} points, max distortion shift=${"%.2f".format(maxShift)} px, " +
                    "max residual after undistortion=${"%.4f".format(maxErr)} px"
        )

        // A kulcspontok float pontossaggal taroltak, ezert nem varhato tokeletes egyezes
        assertTrue("distortion too small to test anything: $maxShift", maxShift > 1.0)
        assertTrue("undistortion residual too high: $maxErr", maxErr < 0.05)
    }
}
