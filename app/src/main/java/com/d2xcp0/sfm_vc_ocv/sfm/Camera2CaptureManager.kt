package com.d2xcp0.sfm_vc_ocv.sfm

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
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
import java.util.*

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
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager

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

        Log.i(tag, "Using cameraId=$cameraId jpegSize=$jpegSize")

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

    private fun chooseBackCamera(manager: android.hardware.camera2.CameraManager): String? {
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
        // Pick a moderate fixed size. Keep this unchanged after calibration.
        // Logcat will show the selected size.
        return sizes
            .filter { it.width >= 1600 && it.height >= 1200 }
            .minByOrNull { it.width * it.height }
            ?: sizes.maxBy { it.width * it.height }
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
        captureRequest.set(CaptureRequest.JPEG_ORIENTATION, 90)

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
}