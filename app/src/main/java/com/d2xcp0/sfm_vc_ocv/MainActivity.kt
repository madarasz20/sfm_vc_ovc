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
import kotlin.math.max
import kotlin.math.pow


class MainActivity : AppCompatActivity() {

    private val CAMERA_PERMISSION_CODE = 100
    private var latestPhotoUri: Uri? = null
    private val savedImages = mutableStateListOf<Uri>()
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (OpenCVLoader.initDebug())
            Log.i("OpenCV", "OpenCV loaded successfully!")
        else
            Log.e("OpenCV", "OpenCV FAILED!")

        setContent {
            var showGallery by remember { mutableStateOf(false) }
            var showDebug   by remember { mutableStateOf(false) }

            when {
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
                else -> {
                    MainScreen(
                        onOpenGallery    = { showGallery = true },
                        onOpenDebug      = { showDebug = true },
                        onOpenCamera     = { if (checkCameraPermission()) openCamera() else requestCameraPermission() },
                        onTestImagePaths = { testImagePaths() },
                        onRunSfM         = { runSfM() },
                        onShowSfMResult  = { showSfMResult() },
                        onClearGallery   = { clearGallery() },
                        onExportPointCloud = { exportPointCloud() },
                        onCalibrate      = { runCalibration() }
                    )
                }
            }
        }

        val loaded = CalibrationStorage.load(this)
        if (loaded != null) {
            K = loaded.first
            D = loaded.second
            Log.i("CALIB", "Loaded calibration: K=${K?.dump()}")
        } else {
            Log.w("CALIB", "No calibration found!")
        }
    }

    private fun runCalibration() {
        Thread {
            val calibrator = CameraCalibrator(this)
            val imgs = calibrator.loadCalibrationImages()

            if (imgs.isEmpty()) {
                runOnUiThread {
                    Toast.makeText(this, "No calibration images found!", Toast.LENGTH_LONG).show()
                }
                return@Thread
            }

            val ok = calibrator.calibrate(imgs)

            if (ok) {
                // FIX 1: Reload K and D immediately after calibration so the
                // next SfM run uses the freshly computed values without restart.
                val reloaded = CalibrationStorage.load(this)
                if (reloaded != null) {
                    K = reloaded.first
                    D = reloaded.second
                    Log.i("CALIB", "Calibration reloaded into memory")
                }
            }

            runOnUiThread {
                if (ok) Toast.makeText(this, "Calibration SUCCESS", Toast.LENGTH_LONG).show()
                else    Toast.makeText(this, "Calibration FAILED",  Toast.LENGTH_LONG).show()
            }
        }.start()
    }

    private fun runSfM() {
        if (savedImages.size < 2) {
            Toast.makeText(this, "Need at least 2 images!", Toast.LENGTH_SHORT).show()
            return
        }

        Thread {
            try {
                val calibPair = CalibrationStorage.load(this)
                if (calibPair == null) {
                    runOnUiThread {
                        Toast.makeText(this, "No calibration found!", Toast.LENGTH_LONG).show()
                    }
                    return@Thread
                }
                val K = calibPair.first
                val D = calibPair.second

                val rawImgs     = savedImages.map { uri -> uriToMat(uri) }
                val resizedImgs = rawImgs.map { img -> resizeForSfM(img) }
                val imgs        = resizedImgs.map { img ->
                    val und = Mat()
                    Calib3d.undistort(img, und, K, D)
                    und
                }

                Log.i("SfM", "Loaded ${imgs.size} images")

                val extractor     = FeatureExtractor()
                val matcher       = FeatureMatcher()
                val poseEstimator = PoseEstimator(K)
                val triangulator  = Triangulator(K)
                val poseRefiner   = PoseRefiner(K, D)
                val anchorMatcher = AnchorMatcher()

                Log.i("SfM", "Extracting features for all frames...")
                val allFeatures = imgs.map { img -> extractor.compute(img) }

                val bestPair = findBestInitialPair(imgs, allFeatures, matcher)
                Log.i("SfM", "Best initial pair = $bestPair-${bestPair + 1}")

                val rotations    = mutableListOf<Mat>()
                val translations = mutableListOf<Mat>()
                rotations.add(Mat.eye(3, 3, CvType.CV_64F))
                translations.add(Mat.zeros(3, 1, CvType.CV_64F))

                val allPoints = mutableListOf<Point3>()

                // BA observation map: pointIndex -> list of (frameIndex, 2D point)
                // Built during the SfM loop, consumed by BA at the end.
                val observations = mutableMapOf<Int, MutableList<Pair<Int, Point>>>()
                var globalPointIndex = 0

                var anchorCloud: List<Point3>? = null
                var anchorDescriptors: Mat?    = null

                for (i in 0 until imgs.size - 1) {

                    val (kp1, desc1) = allFeatures[i]
                    val (kp2, desc2) = allFeatures[i + 1]

                    val matches = matcher.match(desc1, desc2, kp1, kp2)
                    Log.i("SfM", "Pair $i-${i+1}: ${matches.size} matches")

                    if (matches.size < 20) {
                        Log.w("SfM", "Skipping pair $i-${i+1}: too few matches (${matches.size})")
                        Log.w("SfM_DIAG", "Pair $i SKIPPED: only ${matches.size} matches")

                        continue
                    }

                    val (Rrel, trel) = poseEstimator.estimatePose(matches)





                    Log.i("SfM_DIAG", "Pair $i pose: R=${Rrel.dump()}, t=${trel.dump()}")

                    val Rprev = rotations.last()
                    val tprev = translations.last()

                    val Rglobal = Mat()
                    Core.gemm(Rprev, Rrel, 1.0, Mat(), 0.0, Rglobal)

                    // After estimatePose, log the RELATIVE rotation angle (this should be 10-15°)
                    val relAngle = rotationAngleDeg(Rrel)

// After computing Rglobal, log the GLOBAL rotation angle (this shows drift)
                    val globalAngle = rotationAngleDeg(Rglobal)

                    Log.i("SfM_DIAG", "Pair $i: relRot=${relAngle.toInt()}° globalRot=${globalAngle.toInt()}°")

                    if (relAngle > 25.0) {
                        Log.w("SfM_DIAG", "Pair $i SKIPPED: relative rotation too large (${relAngle.toInt()}°)")
                        continue
                    }

                    val temp = Mat()
                    Core.gemm(Rprev, trel, 1.0, Mat(), 0.0, temp)
                    val tglobal = Mat()
                    Core.add(tprev, temp, tglobal)

                    val coarseCloud = triangulator.triangulate(
                        matches, Rprev, tprev, Rglobal, tglobal
                    )
                    Log.i("SfM", "Pair $i coarse triangulation: ${coarseCloud.size} points")

                    if (coarseCloud.isEmpty()) {
                        Log.w("SfM", "No 3D points for pair $i, skipping")
                        Log.w("SfM_DIAG", "Pair $i SKIPPED: empty coarse cloud")
                        continue
                    }

                    if ((anchorCloud == null || i == bestPair) && coarseCloud.size > 30) {
                        anchorCloud       = coarseCloud
                        anchorDescriptors = desc1.clone()
                        Log.i("SfM", "Anchor cloud set at pair $i (${anchorCloud!!.size} pts)")
                    }

                    // Get the inlier 2D points for frame i+1 before pose refinement
                    // (matches may be updated by estimatePose — capture here)
                    val (pts1matched, pts2matched) = matches.getMatchedPoints()

                    var (refinedCloud, Rref, tref) = poseRefiner.refine(
                        coarseCloud, pts2matched, Rglobal, tglobal
                    )

                    Log.i("SfM_DIAG", "Pair $i: matches=${matches.size}, " +
                            "coarse=${coarseCloud.size}, refined=${refinedCloud.size}")

                    if (anchorCloud != null && anchorDescriptors != null) {
                        val (anchor3D, anchor2D) = anchorMatcher.match3DTo2D(
                            anchorCloud!!, anchorDescriptors!!, kp2, desc2
                        )

                        if (anchor3D.size >= 12) {
                            Log.i("SfM", "Pair $i anchor matches: ${anchor3D.size}")

                            val (refCloudAnchor, RrefAnchor, trefAnchor) =
                                poseRefiner.refine(anchor3D, anchor2D, Rglobal, tglobal)

                            if (translationIsValid(trefAnchor)) {
                                Log.i("SfM", "Pair $i: using anchor-based pose")
                                refinedCloud = refCloudAnchor
                                Rref         = RrefAnchor
                                tref         = trefAnchor
                            }
                        }
                    }

                    if (!translationIsValid(tref)) {
                        Log.w("SfM", "Invalid translation at frame ${i+1} → skipping")
                        Log.w("SfM_DIAG", "Pair $i SKIPPED: invalid translation mag=" +
                                "${Math.sqrt(tref.get(0,0)[0].pow(2) + tref.get(1,0)[0].pow(2) + tref.get(2,0)[0].pow(2))}")
                        continue
                    }

                    rotations.add(Rref)
                    translations.add(tref)

                    // FIX: Capture matched points AFTER all pose updates are done,
                    // and build observations correctly using both frame indices.
                    // Previous version called getMatchedPoints() twice and used
                    // the wrong index (pts2frame instead of pts1matched/pts2matched).
                    val frameIndexLeft  = i
                    val frameIndexRight = rotations.size - 1  // actual index of the new frame

                    for (k in refinedCloud.indices) {
                        val ptIdx = globalPointIndex + k

                        // Clamp k to valid range for both point lists
                        val obs2Didx = minOf(k, pts2matched.size - 1)
                        val obs2D1   = if (k < pts1matched.size) pts1matched[k] else pts1matched.last()
                        val obs2D2   = if (obs2Didx < pts2matched.size) pts2matched[obs2Didx] else pts2matched.last()

                        observations.getOrPut(ptIdx) { mutableListOf() }.apply {
                            add(frameIndexLeft  to obs2D1)
                            add(frameIndexRight to obs2D2)
                        }
                    }

                    globalPointIndex += refinedCloud.size
                    allPoints.addAll(refinedCloud)

                    Log.i("SfM", "Frame ${i+1} added. Total points so far: ${allPoints.size}")
                }

                Log.i("SfM", "Sequential SfM done. Raw points: ${allPoints.size}, " +
                        "frames: ${rotations.size}, observations: ${observations.size}")

                if (allPoints.isEmpty()) {
                    runOnUiThread {
                        Toast.makeText(this, "SfM failed: no points reconstructed", Toast.LENGTH_LONG).show()
                    }
                    return@Thread
                }

                // Bundle adjustment — jointly optimizes all poses and 3D points
                val bundleAdjuster = BundleAdjuster(K, D)
                val baResult = bundleAdjuster.adjust(
                    allPoints.toMutableList(),
                    rotations,
                    translations,
                    observations
                )

                Log.i("SfM", "BA complete: ${baResult.points3D.size} points, " +
                        "final reprojection error=${baResult.finalReprojError}px")

                Log.i("SfM_DIAG", "Pre-BA: ${allPoints.size} pts, " +
                        "observations=${observations.size}, frames=${rotations.size}")
                Log.i("SfM_DIAG", "Post-BA: ${baResult.points3D.size} pts")

                // Warn if BA result looks wrong
                if (baResult.finalReprojError > 5.0) {
                    Log.w("SfM", "High reprojection error after BA (${baResult.finalReprojError}px) " +
                            "— consider more images with better baseline")
                }

                // Statistical outlier removal then normalize for display
                val cleaned    = removeStatisticalOutliers(baResult.points3D)
                val normalized = normalizePointCloud(cleaned)

                Log.i("SfM_DIAG", "Post-outlier: ${cleaned.size} pts")
                Log.i("SfM_DIAG", "Post-normalize: ${normalized.size} pts")

                Log.i("SfM", "Final cloud: ${normalized.size} points " +
                        "(after outlier removal from ${baResult.points3D.size})")

                reconstructedCloud      = normalized
                PointCloudHolder.points = normalized

                runOnUiThread {
                    Toast.makeText(
                        this,
                        "SfM complete: ${normalized.size} points\n" +
                                "Reprojection error: ${"%.2f".format(baResult.finalReprojError)}px",
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

    // FIX 2: Accepts pre-computed features — no redundant extraction
    private fun findBestInitialPair(
        imgs: List<Mat>,
        allFeatures: List<Pair<org.opencv.core.MatOfKeyPoint, Mat>>,
        matcher: FeatureMatcher
    ): Int {
        var bestIndex = 0
        var bestScore = 0.0

        for (i in 0 until imgs.size - 1) {
            val (kp1, d1) = allFeatures[i]
            val (kp2, d2) = allFeatures[i + 1]

            val matches = matcher.match(d1, d2, kp1, kp2)

            DebugVisualizer.saveMatchesImage(
                this, imgs[i], imgs[i+1], kp1, kp2,
                matches.toListOfPairs(), "pair_${i}_${i+1}"
            )

            if (matches.size < 12) continue

            val (pts1, pts2) = matches.getMatchedPoints()
            var sumDisp = 0.0
            for (k in pts1.indices) {
                sumDisp += Math.hypot(pts1[k].x - pts2[k].x, pts1[k].y - pts2[k].y)
            }
            val avgDisp = sumDisp / pts1.size
            val score   = matches.size * avgDisp

            Log.i("SfM", "Pair $i score=${"%.1f".format(score)} " +
                    "(matches=${matches.size}, avgDisp=${"%.1f".format(avgDisp)}px)")

            if (score > bestScore) {
                bestScore = score
                bestIndex = i
            }
        }

        Log.i("SfM", "Best pair: $bestIndex (score=${"%.1f".format(bestScore)})")
        return bestIndex
    }

    // FIX 4: Much wider range — previous 0.001..5.0 was room-scale only.
    // Small object scanning produces translations in the 0.0001..50.0 range.
    private fun translationIsValid(t: Mat): Boolean {
        if (t.empty()) return false
        val tx = t.get(0, 0)[0]
        val ty = t.get(1, 0)[0]
        val tz = t.get(2, 0)[0]

        //return !tx.isNaN() && !ty.isNaN() && !tz.isNaN()

        if (tx.isNaN() || ty.isNaN() || tz.isNaN()) return false

        val mag = Math.sqrt(tx*tx + ty*ty + tz*tz)

        if (mag !in 0.0001..50.0) {
            Log.w("SfM", "Translation magnitude out of range: ${"%.5f".format(mag)}")
            return false
        }
        return true
    }

    private fun normalizePointCloud(points: List<Point3>): List<Point3> {
        if (points.isEmpty()) return points

        var cx = 0.0; var cy = 0.0; var cz = 0.0
        for (p in points) { cx += p.x; cy += p.y; cz += p.z }
        cx /= points.size; cy /= points.size; cz /= points.size

        val centered = points.map { p -> Point3(p.x - cx, p.y - cy, p.z - cz) }

        var maxDist = 0.0
        for (p in centered) {
            val d = Math.sqrt(p.x*p.x + p.y*p.y + p.z*p.z)
            if (d > maxDist) maxDist = d
        }

        if (maxDist < 1e-9) return centered
        val scale = 1.0 / maxDist
        return centered.map { p -> Point3(p.x*scale, p.y*scale, p.z*scale) }
    }

    // Resize to 1200px max — must match CameraCalibrator.TARGET_SIZE exactly
    private fun resizeForSfM(src: Mat): Mat {
        val maxDim = CameraCalibrator.TARGET_SIZE
        val w = src.width().toDouble()
        val h = src.height().toDouble()
        val scale = maxDim / max(w, h)
        if (scale >= 1.0) return src.clone()
        val dst = Mat()
        Imgproc.resize(src, dst, Size(w * scale, h * scale))
        return dst
    }

    // ── Boilerplate below — unchanged ─────────────────────────────────────

    private fun exportPointCloud() {
        if (reconstructedCloud == null) {
            Toast.makeText(this, "No point cloud to export!", Toast.LENGTH_SHORT).show()
            return
        }
        val file = PointCloudExporter.exportPLY(this, reconstructedCloud!!)
        Toast.makeText(this, "Saved to:\n${file?.absolutePath}", Toast.LENGTH_LONG).show()
    }

    private fun clearGallery() {
        val dir = getExternalFilesDir("Pictures")
        dir?.listFiles()?.forEach { it.delete() }
        savedImages.clear()
        Toast.makeText(this, "Gallery cleared!", Toast.LENGTH_LONG).show()
    }

    private fun uriToMat(uri: Uri): Mat {
        val input = contentResolver.openInputStream(uri)
        val bytes = input!!.readBytes()
        val buf   = Mat(1, bytes.size, CvType.CV_8UC1)
        buf.put(0, 0, bytes)
        return Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR)
    }

    private fun showSfMResult() {
        val cloud = reconstructedCloud
        if (cloud == null || cloud.isEmpty()) {
            Toast.makeText(this, "No point cloud to display.", Toast.LENGTH_SHORT).show()
            return
        }
        PointCloudHolder.points = cloud
        startActivity(Intent(this, PointCloudActivity::class.java))
    }

    private fun testImagePaths() {
        val paths = StorageUtils.getAllImageFilePaths(this)
        Toast.makeText(this, "Found ${paths.size} images\n${nativeTest()}", Toast.LENGTH_LONG).show()
    }

    private fun createImageFile(): File {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir = getExternalFilesDir("Pictures")
        return File(storageDir, "IMG_$timestamp.jpg")
    }

    private fun openCamera() {
        val imageFile = createImageFile()
        latestPhotoUri = FileProvider.getUriForFile(this, "${packageName}.provider", imageFile)
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, latestPhotoUri)
        cameraLauncher.launch(cameraIntent)
    }

    private fun checkCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE
        )
    }

    override fun onRequestPermissionsResult(req: Int, p: Array<out String>, g: IntArray) {
        super.onRequestPermissionsResult(req, p, g)
        if (req == CAMERA_PERMISSION_CODE && g.isNotEmpty() &&
            g[0] == PackageManager.PERMISSION_GRANTED) openCamera()
    }

    private fun removeStatisticalOutliers(points: List<Point3>, neighbors: Int = 20, stdRatio: Double = 2.0): List<Point3> {
        if (points.size < neighbors + 1) return points

        // For each point, compute mean distance to k nearest neighbors
        val meanDists = points.map { p ->
            val dists = points
                .filter { it !== p }
                .map { q ->
                    val dx = p.x - q.x; val dy = p.y - q.y; val dz = p.z - q.z
                    Math.sqrt(dx*dx + dy*dy + dz*dz)
                }
                .sorted()
                .take(neighbors)
            dists.average()
        }

        val globalMean = meanDists.average()
        val globalStd  = Math.sqrt(meanDists.map { (it - globalMean) * (it - globalMean) }.average())
        val threshold  = globalMean + stdRatio * globalStd

        val filtered = points.filterIndexed { i, _ -> meanDists[i] < threshold }
        Log.i("SfM", "Outlier removal: ${points.size} → ${filtered.size} points")
        return filtered
    }

    private fun rotationAngleDeg(R: Mat): Double {
        // Rotation angle = arccos((trace(R) - 1) / 2)
        val trace = R.get(0,0)[0] + R.get(1,1)[0] + R.get(2,2)[0]
        val cosAngle = (trace - 1.0) / 2.0
        return Math.toDegrees(Math.acos(cosAngle.coerceIn(-1.0, 1.0)))
    }
}