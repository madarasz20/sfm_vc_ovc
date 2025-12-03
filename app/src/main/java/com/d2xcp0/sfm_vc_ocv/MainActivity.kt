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
import org.opencv.calib3d.Calib3d
import org.opencv.imgproc.Imgproc
import android.graphics.BitmapFactory
import com.d2xcp0.sfm_vc_ocv.screens.DebugGalleryScreen
import org.opencv.android.Utils
import org.opencv.features2d.SIFT
import kotlin.math.max
import kotlin.math.min


class MainActivity : AppCompatActivity() {

    private val CAMERA_PERMISSION_CODE = 100
    private var latestPhotoUri: Uri? = null
    private val savedImages = mutableStateListOf<Uri>()

    //Store 3D reconstruction result
    private var reconstructedCloud: List<Point3>? = null

    private var K: Mat? = null
    private var D: Mat? = null


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
    private fun debugPrintCalibration() {
        val prefs = getSharedPreferences("calib", MODE_PRIVATE)

        val kStr = prefs.getString("K", null)
        val dStr = prefs.getString("D", null)

        Log.i("CALIB_DEBUG", "K = $kStr")
        Log.i("CALIB_DEBUG", "D = $dStr")
    }



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        //Load OpenCV
        if (OpenCVLoader.initDebug())
            Log.i("OpenCV", "OpenCV loaded successfully!")
        else
            Log.e("OpenCV", "OpenCV FAILED!")

        setContent {

            //UI
            var showGallery by remember { mutableStateOf(false) }
            var showDebug by remember { mutableStateOf(false) }

            when {
                //Show images with pairs
                showDebug -> {
                    var debugFiles by remember { mutableStateOf(StorageUtils.loadDebugImages(this)) }

                    DebugGalleryScreen(
                        debugFiles = debugFiles,
                        onBack = { showDebug = false },
                        onClearDebug = {
                            StorageUtils.clearDebugImages(this)
                            debugFiles = StorageUtils.loadDebugImages(this)
                        }
                    )
                }

                showGallery -> {
                    GalleryScreen(
                        onBack = { showGallery = false },
                        images = savedImages.toList()
                    )
                }

                //main ui
                else -> {
                    MainScreen(
                        onOpenGallery = { showGallery = true },
                        onOpenDebug = { showDebug = true },
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

        //load saved calibration
        val loaded = CalibrationStorage.load(this)
        if (loaded != null) {
            K = loaded.first
            D = loaded.second
            Log.i("CALIB", "Loaded calibration")
        } else {
            Log.w("CALIB", "No calibration file found!")
        }
    }


    //camera calibration
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

    private fun exportPointCloud() {
        if (reconstructedCloud == null) {
            Toast.makeText(this, "No point cloud to export!", Toast.LENGTH_SHORT).show()
            return
        }

        val file = PointCloudExporter.exportPLY(this, reconstructedCloud!!)
        Toast.makeText(this, "Point cloud saved to:\n${file?.absolutePath}", Toast.LENGTH_LONG).show()
    }

    private fun clearGallery() {
        val dir = getExternalFilesDir("Pictures")       //saved photos folder
        if (dir != null && dir.exists()) {
            dir.listFiles()?.forEach { file ->
                try {
                    file.delete()
                    Log.i("CLEAR", "Deleted: ${file.absolutePath}")
                } catch (e: Exception) {
                    Log.e("CLEAR", "Failed to delete ${file.absolutePath}", e)
                }
            }
        }

        savedImages.clear()   //clear UI

        Toast.makeText(this, "Gallery cleared. All images deleted!", Toast.LENGTH_LONG).show()
    }

    private fun uriToMat(uri: Uri): Mat {
        val input = contentResolver.openInputStream(uri)
        val bytes = input!!.readBytes()
        val buf = Mat(1, bytes.size, CvType.CV_8UC1)
        buf.put(0, 0, bytes)
        return Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR)
    }

    //SfM pipeline
    private fun runSfM() {
        if (savedImages.size < 2) {
            Toast.makeText(this, "Need at least 2 images!", Toast.LENGTH_SHORT).show()
            return
        }

        Thread {
            try {
                //load calibration
                val calibPair = CalibrationStorage.load(this)
                if (calibPair == null) {
                    runOnUiThread {
                        Toast.makeText(this, "No calibration found!", Toast.LENGTH_LONG).show()
                    }
                    return@Thread
                }
                val K = calibPair.first
                val D = calibPair.second

                // resize and undsitort images
                val rawImgs = savedImages.map { uri -> uriToMat(uri) }
                val resizedImgs = rawImgs.map { img -> resizeForCalibration(img) }

                val imgs = resizedImgs.map { img ->
                    val und = Mat()
                    Calib3d.undistort(img, und, K, D)
                    und
                }

                Log.i("SfM", "Loaded ${imgs.size} images for SfM")

                //initialize moduls
                val extractor = FeatureExtractor()
                val matcher = FeatureMatcher()
                val poseEstimator = PoseEstimator(K)
                val triangulator = Triangulator(K)
                val poseRefiner = PoseRefiner(K)
                val anchorMatcher = AnchorMatcher()

                val bestPair = findBestInitialPair(imgs, extractor, matcher)
                Log.i("SfM", "Best initial pair = $bestPair-${bestPair + 1}")


                val rotations = mutableListOf<Mat>()
                val translations = mutableListOf<Mat>()

                rotations.add(Mat.eye(3, 3, CvType.CV_64F))
                translations.add(Mat.zeros(3, 1, CvType.CV_64F))

                val allPoints = mutableListOf<Point3>()

                //Anchor 3D and descriptors for translation registration
                var anchorCloud: List<Point3>? = null
                var anchorDescriptors: Mat? = null

                //Sequential SfM over adjacent pairs
                for (i in 0 until imgs.size - 1) {

                    //Feature extraction
                    val (kp1, desc1) = extractor.compute(imgs[i])
                    val (kp2, desc2) = extractor.compute(imgs[i + 1])

                    //matching
                    val matches = matcher.match(desc1, desc2, kp1, kp2)
                    Log.i("SfM", "Pair $i-${i + 1}: matches = ${matches.size}")

                    if (matches.size < 20) {
                        Log.w("SfM", "Skipping pair $i-${i + 1}: too few matches")
                        continue
                    }

                    //relative poseestimation
                    val (Rrel, trel) = poseEstimator.estimatePose(matches)

                    val Rprev = rotations.last()
                    val tprev = translations.last()

                    //global pose
                    val Rglobal = Mat()
                    Core.gemm(Rprev, Rrel, 1.0, Mat(), 0.0, Rglobal)

                    val temp = Mat()
                    Core.gemm(Rprev, trel, 1.0, Mat(), 0.0, temp)
                    val tglobal = Mat()
                    Core.add(tprev, temp, tglobal)

                    //triangulation
                    val coarseCloud = triangulator.triangulate(
                        matches,
                        Rprev, tprev,
                        Rglobal, tglobal
                    )
                    Log.i("SfM", "Pair $i coarse triangulation: ${coarseCloud.size} points")

                    if (coarseCloud.isEmpty()) {
                        Log.w("SfM", "No 3D points for pair $i, skipping")
                        continue
                    }

                    //init anchorcloud, then load coarsecloud
                    if ((anchorCloud == null || i == bestPair) && coarseCloud.size > 30) {
                        anchorCloud = coarseCloud
                        anchorDescriptors = desc1.clone()
                        Log.i("SfM", "Anchor cloud set at pair $i with ${anchorCloud!!.size} points")
                    }

                    //pair based  refinement
                    val (_, pts2) = matches.getMatchedPoints()
                    var (refinedCloud, Rref, tref) = poseRefiner.refine(
                        coarseCloud,
                        pts2,
                        Rglobal,
                        tglobal
                    )

                    // anchor based 2d -> 3d recreation
                    if (anchorCloud != null && anchorDescriptors != null) {
                        val (anchor3D, anchor2D) =
                            anchorMatcher.match3DTo2D(anchorCloud!!, anchorDescriptors!!, kp2, desc2)

                        if (anchor3D.size >= 12) {
                            Log.i("SfM", "Pair $i anchor matches: ${anchor3D.size}")

                            val (refCloudAnchor, RrefAnchor, trefAnchor) =
                                poseRefiner.refine(anchor3D, anchor2D, Rglobal, tglobal)

                            if (translationIsValid(trefAnchor)) {
                                Log.i("SfM", "Pair $i: using anchor-based pose refinement")
                                refinedCloud = refCloudAnchor
                                Rref = RrefAnchor
                                tref = trefAnchor
                            }
                        }
                    }

                    //sanity check
                    if (!translationIsValid(tref)) {
                        Log.w("SfM", "Invalid translation at frame ${i + 1} → skipping this frame")
                        continue
                    }

                    //update cameraposes
                    rotations.add(Rref)
                    translations.add(tref)

                    //accumulate 3D points
                    allPoints.addAll(refinedCloud)
                }

                //normalize cloud for display
                val normalized = normalizePointCloud(allPoints)
                reconstructedCloud = normalized
                PointCloudHolder.points = normalized

                runOnUiThread {
                    Toast.makeText(
                        this,
                        "SfM complete: ${normalized.size} points reconstructed!",
                        Toast.LENGTH_LONG
                    ).show()
                }

            } catch (e: Exception) {
                Log.e("SfM", "SfM failed", e)
                runOnUiThread {
                    Toast.makeText(this, "SfM failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun showSfMResult() {
        val cloud = reconstructedCloud
        if (cloud == null || cloud.isEmpty()) {
            Toast.makeText(this, "No point cloud to display.", Toast.LENGTH_SHORT).show()
            return
        }

        PointCloudHolder.points = cloud   // make sure viewer sees latest cloud
        startActivity(Intent(this, PointCloudActivity::class.java))
    }

    //test img filepath (not needed)
    private fun testImagePaths() {
        val paths = StorageUtils.getAllImageFilePaths(this)
        Toast.makeText(this, "Found ${paths.size} images\n${nativeTest()}", Toast.LENGTH_LONG).show()
    }

    //camera and permission
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

    private fun resizeForCalibration(src: Mat): Mat {
        val maxDim = 1200.0
        val w = src.width().toDouble()
        val h = src.height().toDouble()
        val scale = maxDim / max(w, h)

        //If image is already small enough, just a copy
        if (scale >= 1.0) return src.clone()

        val dst = Mat()
        Imgproc.resize(src, dst, Size(w * scale, h * scale))
        return dst
    }

    private fun findBestInitialPair(
        imgs: List<Mat>,
        extractor: FeatureExtractor,
        matcher: FeatureMatcher
    ): Int {
        var bestIndex = 0
        var bestScore = 0.0

        for (i in 0 until imgs.size - 1) {
            val (kp1, d1) = extractor.compute(imgs[i])
            val (kp2, d2) = extractor.compute(imgs[i + 1])

            val matches = matcher.match(d1, d2, kp1, kp2)

            DebugVisualizer.saveMatchesImage(
                this,
                imgs[i],
                imgs[i+1],
                kp1,
                kp2,
                matches.toListOfPairs(),
                "pair_${i}_${i+1}"
            )


            if (matches.size < 12) continue

            val (pts1, pts2) = matches.getMatchedPoints()

            // average 2D displacement (parallax)
            var sumDisp = 0.0
            for (k in pts1.indices) {
                sumDisp += Math.hypot(pts1[k].x - pts2[k].x, pts1[k].y - pts2[k].y)
            }
            val avgDisp = sumDisp / pts1.size

            val score = matches.size * avgDisp
            if (score > bestScore) {
                bestScore = score
                bestIndex = i
            }
        }
        return bestIndex
    }

    private fun translationIsValid(t: Mat): Boolean {
        if (t.empty()) return false
        val tx = t.get(0,0)[0]
        val ty = t.get(1,0)[0]
        val tz = t.get(2,0)[0]

        if (tx.isNaN() || ty.isNaN() || tz.isNaN()) return false

        val mag = Math.sqrt(tx*tx + ty*ty + tz*tz)
        return mag in 0.001..5.0   // tuned for room-sized recon
    }

    private fun normalizePointCloud(points: List<Point3>): List<Point3> {
        if (points.isEmpty()) return points

        //centeroid compute
        var cx = 0.0
        var cy = 0.0
        var cz = 0.0

        for (p in points) {
            cx += p.x
            cy += p.y
            cz += p.z
        }

        cx /= points.size
        cy /= points.size
        cz /= points.size

        //sub
        val centered = points.map { p ->
            Point3(p.x - cx, p.y - cy, p.z - cz)
        }

        //scale computng
        var maxDist = 0.0
        for (p in centered) {
            val d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            if (d > maxDist) maxDist = d
        }

        // zero prevent
        if (maxDist < 1e-9) return centered

        val scale = 1.0 / maxDist

        //scaling the cloud
        val normalized = centered.map { p ->
            Point3(p.x * scale, p.y * scale, p.z * scale)
        }

        return normalized
    }


}
