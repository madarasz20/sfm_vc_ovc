package com.d2xcp0.sfm_vc_ocv.camera

import kotlin.math.sqrt

// A kalibracios szamitasok tiszta (Android- es OpenCV-fuggetlen) fuggvenyei,
// hogy a gepen futo unit tesztekbol (src/test) is ellenorizhetok legyenek.
object CalibrationMath {

    // 35mm film atloja (36x24 mm), az ekvivalens fokusz erre vonatkozik
    const val FILM_DIAGONAL_MM = 43.2666

    // Belso parameterek az EXIF 35mm-ekvivalens fokusztavolsagbol:
    // f_px = f35 / 43,27 * kepatlo_px, fopont a kep kozepen. Visszaad: [fx, fy, cx, cy]
    fun intrinsicsFromExif(f35: Int, width: Int, height: Int): DoubleArray {
        require(f35 > 0) { "f35 must be positive: $f35" }
        val diagPx = sqrt(width.toDouble() * width + height.toDouble() * height)
        val f = f35 / FILM_DIAGONAL_MM * diagPx
        return doubleArrayOf(f, f, width / 2.0, height / 2.0)
    }

    // Android LENS_DISTORTION sorrend: [k1, k2, k3, p1, p2]
    // OpenCV sorrend:                  [k1, k2, p1, p2, k3]
    // null, ha kevesebb mint 5 egyutthato erkezik
    fun androidToOpenCvDistortion(distortion: FloatArray): DoubleArray? {
        if (distortion.size < 5) return null
        return doubleArrayOf(
            distortion[0].toDouble(), // k1
            distortion[1].toDouble(), // k2
            distortion[3].toDouble(), // p1
            distortion[4].toDouble(), // p2
            distortion[2].toDouble()  // k3
        )
    }
}
