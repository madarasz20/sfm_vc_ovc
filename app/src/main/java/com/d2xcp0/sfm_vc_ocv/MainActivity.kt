package com.d2xcp0.sfm_vc_ocv

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.app.ActivityCompat
import com.d2xcp0.sfm_vc_ocv.screens.GalleryScreen
import com.d2xcp0.sfm_vc_ocv.screens.MainScreen
import com.d2xcp0.sfm_vc_ocv.sfm.*
import com.d2xcp0.sfm_vc_ocv.utils.StorageUtils
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import android.graphics.BitmapFactory
import org.opencv.android.Utils


class MainActivity : AppCompatActivity() {

    private val CAMERA_PERMISSION_CODE = 100
    private var latestPhotoUri: Uri? = null
    private val savedImages = mutableStateListOf<Uri>()

    // Store 3D reconstruction result
    private var reconstructedCloud: List<Point3>? = null

    companion object {
        init { System.loadLibrary("native-lib") }
    }
    external fun nativeTest(): String


    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            latestPhotoUri?.let { savedImages.add(it) }
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (OpenCVLoader.initDebug())
            Log.i("OpenCV", "OpenCV loaded successfully!")
        else
            Log.e("OpenCV", "OpenCV FAILED!")

        setContent {
            var showGallery by remember { mutableStateOf(false) }

            if (showGallery) {
                GalleryScreen(
                    onBack = { showGallery = false },
                    images = savedImages.toList()
                )
            } else {
                MainScreen(
                    onOpenGallery = { showGallery = true },
                    onOpenCamera = {
                        if (checkCameraPermission()) openCamera() else requestCameraPermission()
                    },
                    onTestImagePaths = { testImagePaths() },
                    onRunSfM = { runSfM() },
                    onShowSfMResult = { showSfMResult() },
                    onClearGallery = { clearGallery() },
                    onExportPointCloud = { exportPointCloud() },
                    onCalibrate = { runCalibration() }
                )
            }
        }
    }

    // ----------------------------------------------------------------------------------------
    //  CAMERA CALIBRATION (THE ONLY VALID VERSION)
    // ----------------------------------------------------------------------------------------
    private fun runCalibration() {
        Thread {
            val calibrator = CameraCalibrator(
                this
            )

            val imgs = calibrator.loadCalibrationImages()

            if (imgs.isEmpty()) {
                runOnUiThread { Toast.makeText(this, "No calibration images found!", Toast.LENGTH_LONG).show() }
                return@Thread
            }

            val ok = calibrator.calibrate(imgs)

            runOnUiThread {
                if (ok) Toast.makeText(this, "Calibration SUCCESS", Toast.LENGTH_LONG).show()
                else Toast.makeText(this, "Calibration FAILED", Toast.LENGTH_LONG).show()
            }

        }.start()
    }

    // ----------------------------------------------------------------------------------------
    //     SfM POINT CLOUD EXPORT
    // ----------------------------------------------------------------------------------------
    private fun exportPointCloud() {
        if (reconstructedCloud == null) {
            Toast.makeText(this, "No point cloud to export!", Toast.LENGTH_SHORT).show()
            return
        }

        val file = PointCloudExporter.exportPLY(this, reconstructedCloud!!)
        Toast.makeText(this, "Point cloud saved to:\n${file?.absolutePath}", Toast.LENGTH_LONG).show()
    }

    // ----------------------------------------------------------------------------------------
    //     CLEAR GALLERY
    // ----------------------------------------------------------------------------------------
    private fun clearGallery() {
        savedImages.clear()
        Toast.makeText(this, "Gallery cleared!", Toast.LENGTH_SHORT).show()
    }

    // ----------------------------------------------------------------------------------------
    //  URI → Mat helper
    // ----------------------------------------------------------------------------------------
    private fun uriToMat(uri: Uri): Mat {
        val input = contentResolver.openInputStream(uri)
        val bytes = input!!.readBytes()
        val buf = Mat(1, bytes.size, CvType.CV_8UC1)
        buf.put(0, 0, bytes)
        return Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR)
    }

    // ----------------------------------------------------------------------------------------
    //  SFM pipeline
    // ----------------------------------------------------------------------------------------
    private fun runSfM() {
        if (savedImages.size < 2) {
            Toast.makeText(this, "Need at least 2 images!", Toast.LENGTH_SHORT).show()
            return
        }

        val img1 = uriToMat(savedImages[0])
        val img2 = uriToMat(savedImages[1])

        // Load calibrated intrinsics if available
        val K = Mat.eye(3, 3, CvType.CV_64F)

        val extractor = FeatureExtractor()
        val matcher = FeatureMatcher()
        val poseEstimator = PoseEstimator(K)
        val triangulator = Triangulator(K)

        val (kp1, desc1) = extractor.compute(img1)
        val (kp2, desc2) = extractor.compute(img2)

        val matches = matcher.match(desc1, desc2, kp1, kp2)

        val (R, t) = poseEstimator.estimatePose(matches)

        reconstructedCloud = triangulator.triangulate(matches, R, t)

        Toast.makeText(this, "Reconstructed ${reconstructedCloud!!.size} points!", Toast.LENGTH_LONG).show()
    }

    private fun showSfMResult() {
        Toast.makeText(this, "3D Viewer not implemented yet.", Toast.LENGTH_SHORT).show()
    }

    // ----------------------------------------------------------------------------------------
    //      TEST NATIVE
    // ----------------------------------------------------------------------------------------
    private fun testImagePaths() {
        val paths = StorageUtils.getAllImageFilePaths(this)
        Toast.makeText(this, "Found ${paths.size} images\n${nativeTest()}", Toast.LENGTH_LONG).show()
    }

    // ----------------------------------------------------------------------------------------
    //         CAMERA + PERMISSION
    // ----------------------------------------------------------------------------------------
    private fun createImageFile(): File {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir = getExternalFilesDir("Pictures")
        return File(storageDir, "IMG_$timestamp.jpg")
    }

    private fun openCamera() {
        val imageFile = createImageFile()
        latestPhotoUri = FileProvider.getUriForFile(
            this, "${packageName}.provider", imageFile
        )

        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, latestPhotoUri)

        cameraLauncher.launch(cameraIntent)
    }

    private fun checkCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_CODE
        )
    }

    override fun onRequestPermissionsResult(req: Int, p: Array<out String>, g: IntArray) {
        super.onRequestPermissionsResult(req, p, g)
        if (req == CAMERA_PERMISSION_CODE && g.isNotEmpty() && g[0] == PackageManager.PERMISSION_GRANTED)
            openCamera()
    }
}
