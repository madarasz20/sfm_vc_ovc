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
import org.opencv.core.Size

class MainActivity : AppCompatActivity() {

    private val CAMERA_PERMISSION_CODE = 100
    private var latestPhotoUri: Uri? = null
    private val savedImages = mutableStateListOf<Uri>()

    // Store the 3D reconstruction results
    private var reconstructedCloud: List<org.opencv.core.Point3>? = null

    companion object {
        init { System.loadLibrary("native-lib") }
    }

    external fun nativeTest(): String

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            latestPhotoUri?.let { uri ->
                savedImages.add(uri)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (OpenCVLoader.initDebug()) {
            Log.i("OpenCV", "OpenCV loaded successfully!")
        } else {
            Log.e("OpenCV", "OpenCV FAILED!")
        }

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
                    onOpenCamera = { if (checkCameraPermission()) openCamera() else requestCameraPermission() },
                    onTestImagePaths = { testImagePaths() },
                    onRunSfM = { runSfM() },
                    onShowSfMResult = { showSfMResult() },
                    onClearGallery = { clearGallery() }
                )
            }
        }
    }
    private fun clearGallery() {
        savedImages.clear()
        Toast.makeText(this, "Gallery cleared!", Toast.LENGTH_SHORT).show()
    }

    // -----------------------------------------
    //  Helper: Convert URI to OpenCV Mat
    // -----------------------------------------
    private fun uriToMat(uri: Uri): Mat {
        val input = contentResolver.openInputStream(uri)
        val bytes = input!!.readBytes()
        val buf = Mat(1, bytes.size, org.opencv.core.CvType.CV_8U)
        buf.put(0, 0, bytes)
        return Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR)
    }

    // -----------------------------------------
    //  SfM pipeline execution
    // -----------------------------------------
    private fun runSfM() {
        if (savedImages.size < 2) {
            Toast.makeText(this, "Need at least 2 images!", Toast.LENGTH_SHORT).show()
            return
        }

        // Load first two images
        val img1 = uriToMat(savedImages[0])
        val img2 = uriToMat(savedImages[1])

        // Create SfM components
        val calibrator = CameraCalibrator(Size(9.0, 6.0), 0.024)
        val K = Mat.eye(3, 3, org.opencv.core.CvType.CV_64F) // Using identity unless you have calibration images

        val extractor = FeatureExtractor()
        val matcher = FeatureMatcher()
        val poseEstimator = PoseEstimator(K)
        val triangulator = Triangulator(K)

        // Extract features
        val (kp1, desc1) = extractor.compute(img1)
        val (kp2, desc2) = extractor.compute(img2)

        // Match features
        val matchSet = matcher.match(desc1, desc2, kp1, kp2)

        // Estimate camera pose
        val (R, t) = poseEstimator.estimatePose(matchSet)

        // Triangulate to obtain 3D point cloud
        reconstructedCloud = triangulator.triangulate(matchSet, R, t)

        Toast.makeText(this, "Reconstructed ${reconstructedCloud!!.size} 3D points!", Toast.LENGTH_LONG).show()
    }

    // -----------------------------------------
    //  Placeholder for 3D viewer
    // -----------------------------------------
    private fun showSfMResult() {
        Toast.makeText(this, "3D Viewer not implemented yet.", Toast.LENGTH_SHORT).show()
    }

    // TEST: Calls native function
    private fun testImagePaths() {
        val paths = StorageUtils.getAllImageFilePaths(this)
        val nativeMsg = nativeTest()
        Toast.makeText(this, "Found ${paths.size} images\n$nativeMsg", Toast.LENGTH_LONG).show()
    }

    private fun createImageFile(): File {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir = getExternalFilesDir("Pictures")
        return File(storageDir, "IMG_$timestamp.jpg")
    }

    private fun openCamera() {
        val imageFile = createImageFile()
        latestPhotoUri = FileProvider.getUriForFile(
            this, "${applicationContext.packageName}.provider", imageFile
        )

        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, latestPhotoUri)
        cameraIntent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        cameraIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

        if (cameraIntent.resolveActivity(packageManager) != null) {
            cameraLauncher.launch(cameraIntent)
        } else {
            Toast.makeText(this, "No camera app found", Toast.LENGTH_SHORT).show()
        }
    }

    private fun checkCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_CODE
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            openCamera()
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }
    }
}
