package com.d2xcp0.sfm_vc_ocv.camera

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * U9, U10: a kalibracios kepletek ellenorzese.
 * Helyi (JVM) unit tesztek: telefon nelkul, a fejlesztoi gepen futnak (./gradlew :app:testDebugUnitTest).
 * A mert ertekek a standard kimenetre kerulnek "SFM_UNIT" elotaggal (a Gradle riportban lathatok).
 */
class CalibrationMathTest {

    // U9a: a tesztsorozat adatai (28 mm-ekvivalens, 1440x1080) -> fx = fy ≈ 1164,87 px, fopont a kep kozepen
    @Test
    fun exifIntrinsicsForTestImages() {
        val (fx, fy, cx, cy) = CalibrationMath.intrinsicsFromExif(28, 1440, 1080)
        println("SFM_UNIT U9a EXIF 28 mm, 1440x1080: fx=%.4f fy=%.4f cx=%.1f cy=%.1f".format(fx, fy, cx, cy))

        // kepatlo = 1800 px -> 28 / 43,2666 * 1800
        assertEquals(28.0 / 43.2666 * 1800.0, fx, 1e-9)
        assertEquals(1164.87, fx, 0.01)
        assertEquals(fx, fy, 0.0)
        assertEquals(720.0, cx, 0.0)
        assertEquals(540.0, cy, 0.0)
    }

    // U9b: allo kep (1080x1440) ugyanazt a fokusztavolsagot adja, csak a fopont cserelodik
    @Test
    fun exifIntrinsicsPortraitSameFocal() {
        val landscape = CalibrationMath.intrinsicsFromExif(28, 1440, 1080)
        val portrait = CalibrationMath.intrinsicsFromExif(28, 1080, 1440)
        println("SFM_UNIT U9b portrait: fx=%.4f (landscape %.4f), cx=%.1f cy=%.1f"
            .format(portrait[0], landscape[0], portrait[2], portrait[3]))

        assertEquals(landscape[0], portrait[0], 1e-9)
        assertEquals(540.0, portrait[2], 0.0)
        assertEquals(720.0, portrait[3], 0.0)
    }

    // U9c: fele akkora kep -> fele akkora fokusztavolsag pixelben (a K a kepmerettel aranyosan skalazodik)
    @Test
    fun exifIntrinsicsScaleWithResolution() {
        val full = CalibrationMath.intrinsicsFromExif(28, 1440, 1080)
        val half = CalibrationMath.intrinsicsFromExif(28, 720, 540)
        println("SFM_UNIT U9c scaling: full fx=%.4f, half fx=%.4f, ratio=%.6f"
            .format(full[0], half[0], half[0] / full[0]))

        assertEquals(full[0] / 2.0, half[0], 1e-9)
    }

    // U9d: hianyzo (0) EXIF ertek -> hiba, nem csendes rossz K
    @Test(expected = IllegalArgumentException::class)
    fun exifIntrinsicsRejectMissingFocal() {
        CalibrationMath.intrinsicsFromExif(0, 1440, 1080)
    }

    // U10a: Android [k1, k2, k3, p1, p2] -> OpenCV [k1, k2, p1, p2, k3].
    // Pontosan abrazolhato (2 hatvany) ertekek, hogy a float->double atalakitas ne okozzon elterest.
    @Test
    fun distortionReorderedToOpenCvOrder() {
        val android = floatArrayOf(0.5f, 0.25f, 0.125f, -0.0625f, 0.03125f) // k1, k2, k3, p1, p2
        val opencv = CalibrationMath.androidToOpenCvDistortion(android)
        println("SFM_UNIT U10a distortion order: android=${android.toList()} -> opencv=${opencv?.toList()}")

        // elvart: k1, k2, p1, p2, k3
        assertArrayEquals(doubleArrayOf(0.5, 0.25, -0.0625, 0.03125, 0.125), opencv, 0.0)
    }

    // U10b: 5-nel kevesebb egyutthato -> null (a hivo nulla torzitassal folytatja)
    @Test
    fun distortionTooShortReturnsNull() {
        val result = CalibrationMath.androidToOpenCvDistortion(floatArrayOf(0.1f, 0.2f, 0.3f))
        println("SFM_UNIT U10b distortion with 3 coefficients: result=$result")
        assertNull(result)
    }
}
