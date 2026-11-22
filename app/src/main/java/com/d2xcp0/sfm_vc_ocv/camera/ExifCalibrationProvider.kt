package com.d2xcp0.sfm_vc_ocv.camera

import android.content.Context
import android.net.Uri
import android.util.Log
import android.util.Size
import androidx.exifinterface.media.ExifInterface
import org.opencv.core.CvType
import org.opencv.core.Mat

// K becslese a kep EXIF adataibol (35mm-ekvivalens fokusztavolsag).
// Akkor kell, ha a keszulek nem adja meg a LENS_INTRINSIC_CALIBRATION-t,
// vagy ha a kepeket nem ez a keszulek keszitette (pl. assets/test_images).
// Torzitast nem ad, D = 0; foporont a kep kozepen.
object ExifCalibrationProvider {

    private const val TAG = "ExifCalibProvider"

    // A kepletet (f35 / 43,27 * kepatlo) a CalibrationMath.intrinsicsFromExif tartalmazza
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

        val (fx, fy, cx, cy) = CalibrationMath.intrinsicsFromExif(f35, targetWidth, targetHeight)
        val f = fx

        val K = Mat.eye(3, 3, CvType.CV_64F)
        K.put(0, 0, fx)
        K.put(1, 1, fy)
        K.put(0, 2, cx)
        K.put(1, 2, cy)

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
