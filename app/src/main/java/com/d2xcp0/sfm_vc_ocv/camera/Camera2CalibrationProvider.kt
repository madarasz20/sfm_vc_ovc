package com.d2xcp0.sfm_vc_ocv.camera

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log
import android.util.Size
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.abs

data class Camera2CalibrationResult(
    val cameraId: String,
    val K: Mat,
    val D: Mat,
    val sourceSize: Size,
    val targetSize: Size
)

object Camera2CalibrationProvider {

    private const val TAG = "Camera2CalibProvider"

    fun loadBackCameraCalibration(
        context: Context,
        targetWidth: Int,
        targetHeight: Int
    ): Camera2CalibrationResult {
        val cameraManager =
            context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

        val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
            val characteristics = cameraManager.getCameraCharacteristics(id)
            characteristics.get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
        } ?: throw IllegalStateException("No back camera found")

        val characteristics = cameraManager.getCameraCharacteristics(cameraId)

        val intrinsics = characteristics.get(
            CameraCharacteristics.LENS_INTRINSIC_CALIBRATION
        ) ?: throw IllegalStateException(
            "LENS_INTRINSIC_CALIBRATION not available for cameraId=$cameraId"
        )

        val distortion = characteristics.get(
            CameraCharacteristics.LENS_DISTORTION
        )

        val preCorrectionArray = characteristics.get(
            CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE
        )

        val activeArray = characteristics.get(
            CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE
        )

        val sourceRect = preCorrectionArray ?: activeArray
        ?: throw IllegalStateException("No active array size available")

        val sourceWidth = sourceRect.width()
        val sourceHeight = sourceRect.height()

        // A skalazas fekvo (szenzor iranyu) kepet feltetelez; allo kepnel
        // fx/fy es cx/cy felcserelodne, ez rossz K-t adna
        if (targetWidth < targetHeight) {
            Log.w(TAG, "Portrait image ${targetWidth}x$targetHeight — K scaling assumes landscape sensor orientation!")
        }

        val scaleX = targetWidth.toDouble() / sourceWidth.toDouble()
        val scaleY = targetHeight.toDouble() / sourceHeight.toDouble()

        val fx = intrinsics[0].toDouble() * scaleX
        val fy = intrinsics[1].toDouble() * scaleY
        val cx = intrinsics[2].toDouble() * scaleX
        val cy = intrinsics[3].toDouble() * scaleY
        val skew = intrinsics[4].toDouble() * scaleX

        val K = Mat.eye(3, 3, CvType.CV_64F)
        K.put(0, 0, fx)
        K.put(0, 1, skew)
        K.put(0, 2, cx)
        K.put(1, 0, 0.0)
        K.put(1, 1, fy)
        K.put(1, 2, cy)
        K.put(2, 0, 0.0)
        K.put(2, 1, 0.0)
        K.put(2, 2, 1.0)

        // Android LENS_DISTORTION sorrend: [k1, k2, k3, p1, p2]
        // OpenCV sorrend:                  [k1, k2, p1, p2, k3]
        val D = if (distortion != null && distortion.size >= 5) {
            val d = Mat.zeros(1, 5, CvType.CV_64F)
            d.put(0, 0, distortion[0].toDouble()) // k1
            d.put(0, 1, distortion[1].toDouble()) // k2
            d.put(0, 2, distortion[3].toDouble()) // p1
            d.put(0, 3, distortion[4].toDouble()) // p2
            d.put(0, 4, distortion[2].toDouble()) // k3
            d
        } else {
            Log.w(TAG, "LENS_DISTORTION unavailable; using zero distortion")
            Mat.zeros(1, 5, CvType.CV_64F)
        }

        Log.i(TAG, "cameraId=$cameraId")
        Log.i(TAG, "target image size=${targetWidth}x$targetHeight")
        Log.i(TAG, "source size=${sourceWidth}x$sourceHeight")
        Log.i(TAG, "raw intrinsics=${intrinsics.joinToString()}")
        Log.i(TAG, "raw distortion=${distortion?.joinToString()}")
        Log.i(TAG, "scaled K=${K.dump()}")
        Log.i(TAG, "D=${D.dump()}")

        return Camera2CalibrationResult(
            cameraId = cameraId,
            K = K,
            D = D,
            sourceSize = Size(sourceWidth, sourceHeight),
            targetSize = Size(targetWidth, targetHeight)
        )
    }
}