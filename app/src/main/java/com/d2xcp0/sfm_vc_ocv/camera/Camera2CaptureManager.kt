package com.d2xcp0.sfm_vc_ocv.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import android.graphics.Rect
import org.opencv.core.CvType
import org.opencv.core.Mat

class Camera2CaptureManager(
    private val context: Context
) {
    private val tag = "Camera2Capture"

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private var cameraId: String? = null
    private var jpegSize: Size? = null
    private var latestCallback: ((Uri) -> Unit)? = null

    //private var cameraK: Mat? = null
    //private var cameraD: Mat? = null
    private var intrinsicSourceSize: Size? = null

    fun startCamera(textureView: TextureView) {
        startBackgroundThread()

        if (textureView.isAvailable) {
            openCamera(textureView)
        } else {
            textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(
                    surface: SurfaceTexture,
                    width: Int,
                    height: Int
                ) {
                    openCamera(textureView)
                }

                override fun onSurfaceTextureSizeChanged(
                    surface: SurfaceTexture,
                    width: Int,
                    height: Int
                ) = Unit

                override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true

                override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
            }
        }
    }

    private fun openCamera(textureView: TextureView) {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.e(tag, "Camera permission missing")
            return
        }

        cameraId = chooseBackCamera(manager)

        if (cameraId == null) {
            Log.e(tag, "No back camera found")
            return
        }

        val characteristics = manager.getCameraCharacteristics(cameraId!!)
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)

        val jpegSizes = map?.getOutputSizes(ImageFormat.JPEG)?.toList().orEmpty()
        jpegSize = chooseFixedJpegSize(jpegSizes)

// IMPORTANT: call after jpegSize is known
        //extractCameraCalibration(characteristics)

        Log.i(tag, "Using cameraId=$cameraId jpegSize=$jpegSize")
        /*Log.i(tag, "K JPEG = ${getCameraKForJpeg().dump()}")
        Log.i(tag, "D = ${getCameraD().dump()}")*/

        imageReader = ImageReader.newInstance(
            jpegSize!!.width,
            jpegSize!!.height,
            ImageFormat.JPEG,
            2
        )

        imageReader!!.setOnImageAvailableListener({ reader ->
            val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            saveImage(image)
        }, backgroundHandler)

        manager.openCamera(
            cameraId!!,
            object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    createPreviewSession(textureView)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    cameraDevice = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e(tag, "Camera error: $error")
                    camera.close()
                    cameraDevice = null
                }
            },
            backgroundHandler
        )
    }

    private fun chooseBackCamera(manager: CameraManager): String? {
        for (id in manager.cameraIdList) {
            val characteristics = manager.getCameraCharacteristics(id)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)

            if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                return id
            }
        }
        return null
    }

    private fun chooseFixedJpegSize(sizes: List<Size>): Size {
        val preferredWidth = 1280
        val preferredHeight = 960
        val preferredRatio = preferredWidth.toDouble() / preferredHeight.toDouble()

        sizes.forEach {
            Log.i(tag, "Available JPEG size: ${it.width}x${it.height}")
        }

        val exact = sizes.firstOrNull {
            it.width == preferredWidth && it.height == preferredHeight
        }

        if (exact != null) {
            Log.i(tag, "Using exact preferred size: $exact")
            return exact
        }

        val sameRatioReasonable = sizes
            .filter {
                val ratio = it.width.toDouble() / it.height.toDouble()
                abs(ratio - preferredRatio) < 0.02 &&
                        it.width >= 1000 &&
                        it.height >= 700
            }
            .minByOrNull {
                abs(it.width - preferredWidth) +
                        abs(it.height - preferredHeight)
            }

        if (sameRatioReasonable != null) {
            Log.w(tag, "Preferred 1280x960 not found. Using closest 4:3 size: $sameRatioReasonable")
            return sameRatioReasonable
        }

        val reasonable = sizes
            .filter { it.width >= 1000 && it.height >= 700 }
            .minByOrNull { it.width * it.height }

        if (reasonable != null) {
            Log.w(tag, "No 4:3 reasonable size found. Using reasonable fallback: $reasonable")
            return reasonable
        }

        val largest = sizes.maxBy { it.width * it.height }
        Log.w(tag, "No reasonable size found. Using largest available: $largest")
        return largest
    }

    private fun createPreviewSession(textureView: TextureView) {
        val camera = cameraDevice ?: return
        val reader = imageReader ?: return
        val size = jpegSize ?: return

        val texture = textureView.surfaceTexture ?: return
        texture.setDefaultBufferSize(size.width, size.height)

        val previewSurface = Surface(texture)

        val previewRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        previewRequest.addTarget(previewSurface)
        applyStableSettings(previewRequest)

        camera.createCaptureSession(
            listOf(previewSurface, reader.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    session.setRepeatingRequest(
                        previewRequest.build(),
                        null,
                        backgroundHandler
                    )
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Log.e(tag, "Preview configuration failed")
                }
            },
            backgroundHandler
        )
    }

    private fun applyStableSettings(builder: CaptureRequest.Builder) {
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)

        // First safe version: stable API capture, not fully manual yet.
        builder.set(
            CaptureRequest.CONTROL_AF_MODE,
            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
        )
        builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        builder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)

        // No zoom/crop region is set.
        // This keeps the default full sensor crop.
    }

    fun capturePhoto(callback: (Uri) -> Unit) {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return
        val reader = imageReader ?: return

        latestCallback = callback

        val captureRequest = camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
        captureRequest.addTarget(reader.surface)
        applyStableSettings(captureRequest)

        // May need adjustment depending on device orientation.
        captureRequest.set(CaptureRequest.JPEG_ORIENTATION, 90)  //90

        session.capture(
            captureRequest.build(),
            object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    session: CameraCaptureSession,
                    request: CaptureRequest,
                    result: TotalCaptureResult
                ) {
                    val exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
                    val iso = result.get(CaptureResult.SENSOR_SENSITIVITY)
                    val focus = result.get(CaptureResult.LENS_FOCUS_DISTANCE)

                    Log.i(
                        tag,
                        "Captured. size=$jpegSize exposure=$exposure iso=$iso focus=$focus cameraId=$cameraId"
                    )
                }
            },
            backgroundHandler
        )
    }

    private fun saveImage(image: Image) {
        try {
            val buffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)

            val file = createImageFile()

            FileOutputStream(file).use { output ->
                output.write(bytes)
            }

            val uri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.provider",
                file
            )

            Log.i(tag, "Saved image: ${file.absolutePath}")
            latestCallback?.invoke(uri)

        } catch (e: Exception) {
            Log.e(tag, "Failed to save image", e)
        } finally {
            image.close()
        }
    }

    private fun createImageFile(): File {
        val timestamp = SimpleDateFormat(
            "yyyyMMdd_HHmmss_SSS",
            Locale.getDefault()
        ).format(Date())

        val storageDir = context.getExternalFilesDir("Pictures")
        return File(storageDir, "IMG_$timestamp.jpg")
    }

    fun closeCamera() {
        try {
            captureSession?.close()
            captureSession = null

            cameraDevice?.close()
            cameraDevice = null

            imageReader?.close()
            imageReader = null

            stopBackgroundThread()
        } catch (e: Exception) {
            Log.e(tag, "Error closing camera", e)
        }
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("Camera2Background").also {
            it.start()
            backgroundHandler = Handler(it.looper)
        }
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
        } catch (e: InterruptedException) {
            Log.e(tag, "Background thread interrupted", e)
        }

        backgroundThread = null
        backgroundHandler = null
    }
    /*private fun extractCameraCalibration(
        characteristics: CameraCharacteristics
    ): Pair<Mat, Mat> {

        val intrinsics = characteristics.get(
            CameraCharacteristics.LENS_INTRINSIC_CALIBRATION
        ) ?: throw IllegalStateException(
            "Camera2 intrinsics unavailable: LENS_INTRINSIC_CALIBRATION is null"
        )

        val distortion = characteristics.get(
            CameraCharacteristics.LENS_DISTORTION
        )

        val activeArray = characteristics.get(
            CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE
        )

        val fx = intrinsics[0].toDouble()
        val fy = intrinsics[1].toDouble()
        val cx = intrinsics[2].toDouble()
        val cy = intrinsics[3].toDouble()
        val skew = intrinsics[4].toDouble()

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

        val D = if (distortion != null) {
            val d = Mat.zeros(1, distortion.size, CvType.CV_64F)
            for (i in distortion.indices) {
                d.put(0, i, distortion[i].toDouble())
            }
            d
        } else {
            Log.w(tag, "Camera2 distortion unavailable: LENS_DISTORTION is null")
            Mat.zeros(1, 5, CvType.CV_64F)
        }

        cameraK = K
        cameraD = D

        intrinsicSourceSize = activeArray?.let { rect ->
            Size(rect.width(), rect.height())
        }

        Log.i(tag, "Camera2 intrinsics raw = ${intrinsics.joinToString()}")
        Log.i(tag, "Camera2 distortion raw = ${distortion?.joinToString()}")
        Log.i(tag, "Camera2 active/pre-correction array = $activeArray")
        Log.i(tag, "Camera2 K raw = ${K.dump()}")
        Log.i(tag, "Camera2 D raw = ${D.dump()}")

        return K to D
    }
    fun getCameraKForJpeg(): Mat {
        val Kraw = cameraK ?: throw IllegalStateException(
            "cameraK is not initialized. Call extractCameraCalibration() after opening the camera."
        )

        val src = intrinsicSourceSize ?: throw IllegalStateException(
            "intrinsicSourceSize is not initialized."
        )

        val dst = jpegSize ?: throw IllegalStateException(
            "jpegSize is not initialized."
        )

        val scaleX = dst.width.toDouble() / src.width.toDouble()
        val scaleY = dst.height.toDouble() / src.height.toDouble()

        val K = Kraw.clone()

        K.put(0, 0, Kraw.get(0, 0)[0] * scaleX) // fx
        K.put(0, 1, Kraw.get(0, 1)[0] * scaleX) // skew
        K.put(0, 2, Kraw.get(0, 2)[0] * scaleX) // cx

        K.put(1, 0, 0.0)
        K.put(1, 1, Kraw.get(1, 1)[0] * scaleY) // fy
        K.put(1, 2, Kraw.get(1, 2)[0] * scaleY) // cy

        K.put(2, 0, 0.0)
        K.put(2, 1, 0.0)
        K.put(2, 2, 1.0)

        Log.i(tag, "Scaled K for JPEG $dst from source $src = ${K.dump()}")

        return K
    }

    fun getCameraD(): Mat {
        return cameraD?.clone()
            ?: throw IllegalStateException(
                "cameraD is not initialized. Call extractCameraCalibration() after opening the camera."
            )
    }

    fun getJpegSize(): Size {
        return jpegSize ?: throw IllegalStateException(
            "jpegSize is not initialized. Camera has not selected JPEG size yet."
        )
    }
    fun initializeCalibrationOnly() {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

        val id = chooseBackCamera(manager)
            ?: throw IllegalStateException("No back camera found")

        cameraId = id

        val characteristics = manager.getCameraCharacteristics(id)
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)

        val jpegSizes = map?.getOutputSizes(ImageFormat.JPEG)?.toList().orEmpty()
        jpegSize = chooseFixedJpegSize(jpegSizes)

        extractCameraCalibration(characteristics)

        Log.i(tag, "Calibration initialized only. cameraId=$cameraId jpegSize=$jpegSize")
        Log.i(tag, "K JPEG = ${getCameraKForJpeg().dump()}")
        Log.i(tag, "D = ${getCameraD().dump()}")
    }*/
}