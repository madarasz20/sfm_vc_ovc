package com.d2xcp0.sfm_vc_ocv.camera

import android.content.Context
import android.net.Uri
import android.util.Log
import android.util.Size
import androidx.exifinterface.media.ExifInterface
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.sqrt

// K becslese a kep EXIF adataibol (35mm-ekvivalens fokusztavolsag).
// Akkor kell, ha a keszulek nem adja meg a LENS_INTRINSIC_CALIBRATION-t,
// vagy ha a kepeket nem ez a keszulek keszitette (pl. assets/test_images).
// Torzitast nem ad, D = 0; foporont a kep kozepen.
object ExifCalibrationProvider {

    private const val TAG = "ExifCalibProvider"

    // 35mm film atloja (36x24 mm), az ekvivalens fokusz erre vonatkozik
    private const val FILM_DIAGONAL_MM = 43.2666

    fun loadFromImage(
        context: Context,
        uri: Uri,
        targetWidth: Int,
        targetHeight: Int
    ): Camera2CalibrationResult {
        val exif = context.contentResolver.openInputStream(uri)?.use { ExifInterface(it) }
            ?: throw IllegalStateException("Cannot open image for EXIF: $uri")

        val f35 = exif.getAttributeInt(ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM, 0)
        if (f35 <= 0) {
            throw IllegalStateException("EXIF FocalLengthIn35mmFilm missing: $uri")
        }

        val diagPx = sqrt(
            targetWidth.toDouble() * targetWidth + targetHeight.toDouble() * targetHeight
        )
        val f = f35 / FILM_DIAGONAL_MM * diagPx

        val K = Mat.eye(3, 3, CvType.CV_64F)
        K.put(0, 0, f)
        K.put(1, 1, f)
        K.put(0, 2, targetWidth / 2.0)
        K.put(1, 2, targetHeight / 2.0)

        val D = Mat.zeros(1, 5, CvType.CV_64F)

        Log.i(TAG, "EXIF f35=${f35}mm, image=${targetWidth}x$targetHeight → f=${"%.1f".format(f)}px")
        Log.i(TAG, "K=${K.dump()}")

        return Camera2CalibrationResult(
            cameraId = "exif",
            K = K,
            D = D,
            sourceSize = Size(targetWidth, targetHeight),
            targetSize = Size(targetWidth, targetHeight)
        )
    }
}
