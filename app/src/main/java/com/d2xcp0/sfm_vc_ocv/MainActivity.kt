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
import com.d2xcp0.sfm_vc_ocv.utils.StorageUtils
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import android.util.Log
import org.opencv.android.OpenCVLoader
import android.widget.TextView
import com.d2xcp0.sfm_vc_ocv.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private val CAMERA_PERMISSION_CODE = 100
    private var latestPhotoUri: Uri? = null
    private val savedImages = mutableStateListOf<Uri>() // Compose-friendly list

    companion object {
        init {
            System.loadLibrary("native-lib")
        }
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
            Log.e("OpenCV","OpenCV FAILED!" )
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
                    onOpenCamera = {
                        if (checkCameraPermission()) openCamera()
                        else requestCameraPermission()
                    },
                    onTestImagePaths = { testImagePaths() }   // ← ADDED CALLBACK
                )
            }
        }
    }

    /** Called by the MainScreen test button */
    private fun testImagePaths() {
        val paths = StorageUtils.getAllImageFilePaths(this)
        val nativeMsg = nativeTest()  // 🔥 call into C++

        Toast.makeText(
            this,
            "Found ${paths.size} images\n$nativeMsg",
            Toast.LENGTH_LONG
        ).show()}

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
