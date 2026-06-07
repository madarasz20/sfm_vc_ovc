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
import com.d2xcp0.sfm_vc_ocv.camera.Camera2CalibrationProvider
import com.d2xcp0.sfm_vc_ocv.camera.Camera2CaptureManager
import com.d2xcp0.sfm_vc_ocv.camera.CameraCalibrator
import com.d2xcp0.sfm_vc_ocv.pointcloud.PointCloudExporter
import com.d2xcp0.sfm_vc_ocv.pointcloud.PointCloudHolder
import com.d2xcp0.sfm_vc_ocv.helper.TrackTriangulator
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.core.*
import org.opencv.calib3d.Calib3d
import org.opencv.imgproc.Imgproc
import com.d2xcp0.sfm_vc_ocv.screens.CameraPreviewScreen
import com.d2xcp0.sfm_vc_ocv.screens.DebugGalleryScreen
import kotlin.math.max
import kotlin.math.pow


class MainActivity : AppCompatActivity() {

    private val CAMERA_PERMISSION_CODE = 100
    private var latestPhotoUri: Uri? = null
    private val savedImages = mutableStateListOf<Uri>()
    private var reconstructedCloud: List<Point3>? = null
    private var K: Mat? = null
    private var D: Mat? = null
    private lateinit var camera2CaptureManager: Camera2CaptureManager

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
        //camera2CaptureManager = Camera2CaptureManager(this)
        //camera2CaptureManager.initializeCalibrationOnly()

        if (OpenCVLoader.initDebug())
            Log.i("OpenCV", "OpenCV loaded successfully!")
        else
            Log.e("OpenCV", "OpenCV FAILED!")

        setContent {
            var showGallery by remember { mutableStateOf(false) }
            var showDebug   by remember { mutableStateOf(false) }
            var showCamera by remember { mutableStateOf(false) }

            when {
                showCamera -> {
                    CameraPreviewScreen(
                        onPhotoCaptured = { uri ->
                            savedImages.add(uri)
                            showCamera = false
                        },
                        onBack = {
                            showCamera = false
                        }
                    )
                }
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
                        onOpenGallery = { showGallery = true },
                        onOpenDebug = { showDebug = true },

                        onOpenCamera = {
                            if (checkCameraPermission()) {
                                showCamera = true
                            } else {
                                requestCameraPermission()
                            }
                        },

                        onTestImagePaths = { testImagePaths() },
                        onRunSfM = { runSfM() },
                        onShowSfMResult = { showSfMResult() },
                        onClearGallery = { clearGallery() },
                        onExportPointCloud = { exportPointCloud() },
                        onCalibrate = { runCalibration() },
                        onSaveSession = {saveSession()},
                        onLoadSession = {loadSession()}
                    )
                }
            }
        }

        /*val loaded = CalibrationStorage.load(this)
        if (loaded != null) {
            //K = loaded.first
            //D = loaded.second
            val K = Mat(3, 3, CvType.CV_64F)
            K.put(
                0, 0,
                948.000064, 0.0,        640.0,
                0.0,        948.000064, 480.0,
                0.0,        0.0,        1.0
            )

            val D = Mat(1, 5, CvType.CV_64F)
            D.put(
                0, 0,
                0.0, 0.0, 0.0, 0.0, 0.0
            )
            Log.i("CALIB", "Loaded calibration: K=${K?.dump()}")
        } else {
            Log.w("CALIB", "No calibration found!")
        }*/
        /*val K = Mat(3, 3, CvType.CV_64F)
        K.put(
            0, 0,
            948.000064, 0.0,        640.0,
            0.0,        948.000064, 480.0,
            0.0,        0.0,        1.0
        )

        val D = Mat(1, 5, CvType.CV_64F)
        D.put(
            0, 0,
            0.0, 0.0, 0.0, 0.0, 0.0
        )
        Log.i("CALIB", "Loaded calibration: K=${K?.dump()}")*/
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
        //legalabb 2 kepet kellett csinalni hogy lehessenfuttatni
        if (savedImages.size < 2) {
            Toast.makeText(this, "Need at least 2 images!", Toast.LENGTH_SHORT).show()
            return
        }

        Thread {
            try {
                //kepek:
                val rawImgs = savedImages.map { uri -> uriToMat(uri) }

                //egyelore elso 3
                /*val rawImgs = savedImages
                    .take(6)        //5el talan
                    .map { uri -> uriToMat(uri) }
                Log.i("SFM_TEST", "Two-image test enabled: rawImgs=${rawImgs.size}")*/

                rawImgs.forEachIndexed { idx, img ->
                    Log.i("SFM", "Raw Images size raw[$idx] = ${img.cols()} x ${img.rows()}")
                }

                val firstImg = rawImgs.first()

                //kamera kalibracio
                val calib = Camera2CalibrationProvider.loadBackCameraCalibration(
                    context = this,
                    targetWidth = firstImg.cols(),
                    targetHeight = firstImg.rows()
                )

                val K = calib.K
                val D = calib.D

                Log.i("SFM", "Camera2 calibration K=${K.dump()}")
                Log.i("SFM", "Camera2 calibration D=${D.dump()}")

                val imgs = rawImgs

                //peldanyostunk OK
                val extractor     = FeatureExtractor()
                val matcher       = FeatureMatcher()
                val poseEstimator = PoseEstimator(K)
                val triangulator  = Triangulator(K)
                val poseRefiner   = PoseRefiner(K, D)
                //val anchorMatcher = AnchorMatcher()
                val trackBuilder = TrackBuilder()

                // minden keprol jellemzo kivon
                Log.i("SfM", "Extracting features for all frames...")
                val allFeatures = imgs.map { img -> extractor.compute(img) }

                //legjobb par kivalaszt Miert?
                //val bestPair = findBestInitialPair(imgs, allFeatures, matcher)
                //Log.i("SfM", "Best initial pair = $bestPair-${bestPair + 1}")

                //identity matrix cretion
                val rotations    = mutableListOf<Mat>()
                val translations = mutableListOf<Mat>()
                rotations.add(Mat.eye(3, 3, CvType.CV_64F))
                translations.add(Mat.zeros(3, 1, CvType.CV_64F))

                val allPoints = mutableListOf<Point3>()

                // BA observation map: pointIndex -> list of (frameIndex, 2D point)
                // Built during the SfM loop, consumed by BA at the end.
                val observations = mutableMapOf<Int, MutableList<Pair<Int, Point>>>()
                var globalPointIndex = 0

                //var anchorCloud: List<Point3>? = null
                //var anchorDescriptors: Mat?    = null

                //kepenkent parositunk
                for (i in 0 until imgs.size - 1) {

                    val (kp1, desc1) = allFeatures[i]
                    val (kp2, desc2) = allFeatures[i + 1]

                    val matches = matcher.match(desc1, desc2, kp1, kp2)
                    Log.i("SfM", "Pair $i-${i+1}: ${matches.size} matches")

                    //ez lehet nagyon lecsokkenti a parokat, miert kell
                    if (matches.size < 20) {
                        Log.w("SfM", "Skipping pair $i-${i+1}: too few matches (${matches.size})")
                        Log.w("SfM", "Pair $i SKIPPED: only ${matches.size} matches")

                        continue
                    }

                    //R,T eloallit
                    //kulso kamera parameterek: relativ rotation es realtive translation
                    val (Rrel, trel) = poseEstimator.estimatePose(matches)

                    Log.i(
                        "SFM_POSE_CHAIN",
                        "pair $i-${i+1} trel=[" +
                                "${"%.3f".format(trel.get(0,0)[0])}, " +
                                "${"%.3f".format(trel.get(1,0)[0])}, " +
                                "${"%.3f".format(trel.get(2,0)[0])}]"
                    )


                    trackBuilder.addPairMatches(i, i + 1, matches)

                    DebugVisualizer.saveMatchesImage(
                        this,
                        imgs[i],
                        imgs[i + 1],
                        kp1,
                        kp2,
                        matches.toListOfPairs(),
                        "pair_${i}_${i+1}_01_pose_inliers"
                    )


                    Log.i("SfM", "Pose estimation: Pair $i pose: R=${Rrel.dump()}, t=${trel.dump()}")

                    //az utolso pozt veszi csak figyelembe?
                    val Rprev = rotations.last()
                    val tprev = translations.last()

                    val Rglobal = Mat()
                    Core.gemm(Rrel, Rprev, 1.0, Mat(), 0.0, Rglobal)

                    // After estimatePose, log the RELATIVE rotation angle (this should be 10-15°)
                    val relAngle = rotationAngleDeg(Rrel)

                    // After computing Rglobal, log the GLOBAL rotation angle (this shows drift)
                    val globalAngle = rotationAngleDeg(Rglobal)

                    Log.i("SfM", "Global Rotation: Pair $i: relRot=${relAngle.toInt()}° globalRot=${globalAngle.toInt()}°")

                    if (relAngle > 25.0) {
                        Log.w("SfM", "Skipping if relative rotation to large : Pair $i SKIPPED: relative rotation too large (${relAngle.toInt()}°)")
                        break //continue
                    }

                    val temp = Mat()
                    Core.gemm(Rrel, tprev, 1.0, Mat(), 0.0, temp)
                    val tglobal = Mat()
                    Core.add(temp, trel, tglobal)

                    Log.i(
                        "SFM_POSE_CHAIN",
                        "frame ${i+1} tglobal=[" +
                                "${"%.3f".format(tglobal.get(0,0)[0])}, " +
                                "${"%.3f".format(tglobal.get(1,0)[0])}, " +
                                "${"%.3f".format(tglobal.get(2,0)[0])}]"
                    )

                    //triangulacio globalis rotation és translation matrixxal

                    val tri = triangulator.triangulate(matches, Rprev, tprev, Rglobal, tglobal)

                    val coarseCloud = tri.points3D
                    val pts1Triangulated = tri.points2DLeft
                    val pts2Triangulated = tri.points2DRight

                    Log.i("SFM_PAIRING", "tri 3D=${coarseCloud.size}")
                    Log.i("SFM_PAIRING", "tri 2D left=${pts1Triangulated.size}")
                    Log.i("SFM_PAIRING", "tri 2D right=${pts2Triangulated.size}")
                    Log.i("SFM_PAIRING", "tri matchIndices=${tri.matchIndices.take(10)}")

                    var (refinedCloud, Rref, tref) = poseRefiner.refine(
                        coarseCloud,
                        pts2Triangulated,
                        Rglobal,
                        tglobal
                    )

                    //val (pts1matched, pts2matched) = matches.getMatchedPoints()
                    Log.i("SfM", "Coarse and Refined cloud size: Pair $i: matches=${matches.size}, " +
                            "coarse=${coarseCloud.size}, refined=${refinedCloud.size}")


                    if (!translationIsValid(tref)) {
                        Log.w("SfM", "Invalid translation at frame ${i+1} → skipping")
                        Log.w("SfM", "Pair $i SKIPPED: invalid translation mag=" +
                                "${Math.sqrt(tref.get(0,0)[0].pow(2) + tref.get(1,0)[0].pow(2) + tref.get(2,0)[0].pow(2))}")
                        break//continue
                    }

                    rotations.add(Rref)
                    translations.add(tref)

                    // FIX: Capture matched points AFTER all pose updates are done,
                    // and build observations correctly using both frame indices.
                    // Previous version called getMatchedPoints() twice and used
                    // the wrong index (pts2frame instead of pts1matched/pts2matched).
                    val frameIndexLeft  = i
                    val frameIndexRight = rotations.size - 1  // actual index of the new frame

                    /*for (k in refinedCloud.indices) {
                        val ptIdx = globalPointIndex + k

                        // Clamp k to valid range for both point lists
                        val obs2Didx = minOf(k, pts2matched.size - 1)
                        val obs2D1   = if (k < pts1matched.size) pts1matched[k] else pts1matched.last()
                        val obs2D2   = if (obs2Didx < pts2matched.size) pts2matched[obs2Didx] else pts2matched.last()

                        observations.getOrPut(ptIdx) { mutableListOf() }.apply {
                            add(frameIndexLeft  to obs2D1)
                            add(frameIndexRight to obs2D2)
                        }
                    }*/

                    if (refinedCloud.size != tri.points.size) {
                        Log.e(
                            "SFM_BA_INPUT",
                            "Cannot safely build observations: refinedCloud.size=${refinedCloud.size}, tri.points.size=${tri.points.size}"
                        )
                        break //continue
                    }

                    /*for (k in refinedCloud.indices) {
                        if (k >= tri.points.size) {
                            Log.e("SFM_BA_INPUT", "refinedCloud larger than tri.points at k=$k")
                            break
                        }

                        val ptIdx = globalPointIndex + k
                        val tp = tri.points[k]

                        observations.getOrPut(ptIdx) { mutableListOf() }.apply {
                            add(frameIndexLeft to tp.point2DLeft)
                            add(frameIndexRight to tp.point2DRight)
                        }
                    }*/
                    for (k in refinedCloud.indices) {
                        val ptIdx = globalPointIndex + k
                        val tp = tri.points[k]

                        observations.getOrPut(ptIdx) { mutableListOf() }.apply {
                            add(frameIndexLeft to tp.point2DLeft)
                            add(frameIndexRight to tp.point2DRight)
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

                Log.i("SFM_TRACK_CLOUD", "MARKER A: reached after pair loop")

                trackBuilder.logStats()

                val tracks = trackBuilder.getTracks()
                val goodTracks = trackBuilder.getGoodTracks(minLength = 3)

                Log.i("SFM_TRACKS", "goodTracks len>=3=${goodTracks.size}")
                Log.i("SFM_TRACK_CLOUD", "MARKER B: before triangulation, goodTracks=${goodTracks.size}")


                val trackTriangulator = TrackTriangulator(K)

                val trackPoints = mutableListOf<Point3>()
                val trackPointTracks = mutableListOf<FeatureTrack>()
                var failedTracks = 0

                for (track in goodTracks) {
                    val p: Point3? = trackTriangulator.triangulateTrack(
                        track,
                        rotations,
                        translations
                    )


                    if (p != null) {
                        trackPoints.add(p)
                        trackPointTracks.add(track)
                    } else {
                        failedTracks++
                    }
                }

                Log.i(
                    "SFM_TRACK_CLOUD",
                    "triangulated track points=${trackPoints.size}/${goodTracks.size}, failed=$failedTracks"
                )

                val trackErrors = mutableListOf<Double>()
                val keptTrackFrameHist = mutableMapOf<String, Int>()

                for (idx in trackPoints.indices) {
                    val err = computeTrackReprojectionError(
                        point3D = trackPoints[idx],
                        track = trackPointTracks[idx],
                        rotations = rotations,
                        translations = translations,
                        K = K,
                        D = D
                    )

                    if (err.isFinite()&& err < 5.0) {
                        trackErrors.add(err)
                        val obs = trackPointTracks[idx].observations.sortedBy { it.frameIndex }
                        val key = "${obs.first().frameIndex}-${obs.last().frameIndex}"
                        keptTrackFrameHist[key] = (keptTrackFrameHist[key] ?: 0) + 1
                    }
                }

                if (trackErrors.isNotEmpty()) {
                    val sorted = trackErrors.sorted()
                    val median = sorted[sorted.size / 2]
                    val avg = trackErrors.average()
                    val max = sorted.last()

                    Log.i(
                        "SFM_TRACK_REPROJ",
                        "track reproj error: count=${trackErrors.size}, " +
                                "avg=${"%.2f".format(avg)} px, " +
                                "median=${"%.2f".format(median)} px, " +
                                "max=${"%.2f".format(max)} px"
                    )
                }

                val filteredTrackPoints = mutableListOf<Point3>()

                for (idx in trackPoints.indices) {
                    val err = computeTrackReprojectionError(
                        point3D = trackPoints[idx],
                        track = trackPointTracks[idx],
                        rotations = rotations,
                        translations = translations,
                        K = K,
                        D = D
                    )

                    if (err < 7.0) {
                        filteredTrackPoints.add(trackPoints[idx])
                    }
                }

                Log.i(
                    "SFM_TRACK_CLOUD",
                    "filtered track cloud=${filteredTrackPoints.size}/${trackPoints.size} using reproj<8px"
                )



                // Bundle adjustment — jointly optimizes all poses and 3D points
                val bundleAdjuster = BundleAdjuster(K, D)
                val useBundleAdjustment = false
                val useTrackCloud = true

                val totalObs = observations.values.sumOf { it.size }

                val badPointKeys = observations.keys.count { it < 0 || it >= allPoints.size }

                val badFrameObs = observations.values.flatten().count { (frameIdx, _) ->
                    frameIdx < 0 || frameIdx >= rotations.size
                }

                val obsCounts = observations.mapValues { it.value.size }

                Log.i("SFM_BA_INPUT", "allPoints=${allPoints.size}")
                Log.i("SFM_BA_INPUT", "rotations=${rotations.size}")
                Log.i("SFM_BA_INPUT", "translations=${translations.size}")
                Log.i("SFM_BA_INPUT", "pointsWithObs=${observations.size}")
                Log.i("SFM_BA_INPUT", "totalObs=$totalObs")

                Log.i(
                    "SFM_BA_INPUT",
                    "obsPerPoint min=${obsCounts.values.minOrNull()} " +
                            "max=${obsCounts.values.maxOrNull()} " +
                            "avg=${obsCounts.values.average()}"
                )

                Log.i("SFM_BA_INPUT", "badPointKeys=$badPointKeys badFrameObs=$badFrameObs")

                val pointsNotExactly2Obs = obsCounts.count { it.value != 2 }

                Log.i("SFM_BA_INPUT", "pointsNotExactly2Obs=$pointsNotExactly2Obs")

                if (badPointKeys != 0 || badFrameObs != 0 || pointsNotExactly2Obs != 0) {
                    Log.e("SFM_BA_INPUT", "Invalid BA input graph. Aborting before BA.")
                    runOnUiThread {
                        Toast.makeText(this, "Invalid BA input graph — check logs", Toast.LENGTH_LONG).show()
                    }
                    return@Thread
                }

                /*val baResult = bundleAdjuster.adjust(
                    allPoints.toMutableList(),
                    rotations,
                    translations,
                    observations
                )*/

                var finalPoints: List<Point3> = emptyList()
                var finalReprojError = -1.0

                var baResult: BundleAdjuster.BAResult? = null

                if (useBundleAdjustment) {
                    val bundleAdjuster = BundleAdjuster(K, D)

                    baResult = bundleAdjuster.adjust(
                        allPoints.toMutableList(),
                        rotations,
                        translations,
                        observations
                    )

                    Log.i(
                        "SfM",
                        "BA complete: ${baResult.points3D.size} points, " +
                                "final reprojection error=${baResult.finalReprojError}px"
                    )

                    finalPoints = baResult.points3D
                    finalReprojError = baResult.finalReprojError
                } else {
                    Log.i("SfM", "BA disabled for first validation run")

                    finalPoints = if (useTrackCloud) {
                        Log.i("SFM_TRACK_CLOUD", "Using track cloud for display: ${trackPoints.size} points")
                        trackPoints.toList()
                        filteredTrackPoints
                    } else {
                        Log.i("SFM_TRACK_CLOUD", "Using old pairwise cloud for display: ${allPoints.size} points")
                        allPoints.toList()
                    }

                    finalReprojError = -1.0
                }

                /*Log.i("SfM", "BA complete: ${baResult.points3D.size} points, " +
                        "final reprojection error=${baResult.finalReprojError}px")

                Log.i("SfM", "Pre-BA: ${allPoints.size} pts, " +
                        "observations=${observations.size}, frames=${rotations.size}")
                Log.i("SfM", "Post-BA: ${baResult.points3D.size} pts")

                // Warn if BA result looks wrong
                if (baResult.finalReprojError > 5.0) {
                    Log.w("SfM", "High reprojection error after BA (${baResult.finalReprojError}px) " +
                            "— consider more images with better baseline")
                }*/

                // Statistical outlier removal then normalize for display
                //val cleaned = removeStatisticalOutliers(finalPoints)
                //val normalized = normalizePointCloud(cleaned)
                val cleaned = trackPoints//finalPoints
                val normalized = normalizePointCloud(cleaned)

                Log.i("SFM_FINAL_INPUT", "finalPoints before cleanup=${finalPoints.size}")
                Log.i("SFM_FINAL_INPUT", "normalized=${normalized.size}")

                Log.i("SfM", "Post-outlier: ${cleaned.size} pts")
                Log.i("SfM", "Post-normalize: ${normalized.size} pts")

                Log.i("SfM", "Final cloud: ${normalized.size} points " +
                        "(after outlier removal from ${baResult?.points3D?.size})")


                reconstructedCloud      = normalized
                PointCloudHolder.points = normalized

                runOnUiThread {
                    Toast.makeText(
                        this,
                        "SfM complete: ${normalized.size} points\n" +
                                "Reprojection error: ${"%.2f".format(baResult?.finalReprojError)}px",
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

            if (matches.size < 12) break//continue

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

    /*private fun uriToMat(uri: Uri): Mat {
        val input = contentResolver.openInputStream(uri)
        val bytes = input!!.readBytes()
        val buf   = Mat(1, bytes.size, CvType.CV_8UC1)
        buf.put(0, 0, bytes)
        return Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR)
    }*/
    private fun uriToMat(uri: Uri): Mat {
        val input = contentResolver.openInputStream(uri)!!
        val bytes = input.readBytes()
        input.close()

        val buf = Mat(1, bytes.size, CvType.CV_8UC1)
        buf.put(0, 0, bytes)
        val mat = Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR)

        // Read EXIF orientation and rotate accordingly
        val exifInput = contentResolver.openInputStream(uri)!!
        val exif = androidx.exifinterface.media.ExifInterface(exifInput)
        exifInput.close()

        val orientation = exif.getAttributeInt(
            androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
            androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL
        )

        val rotated = when (orientation) {
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90  -> {
                val dst = Mat()
                Core.rotate(mat, dst, Core.ROTATE_90_CLOCKWISE)
                dst
            }
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 -> {
                val dst = Mat()
                Core.rotate(mat, dst, Core.ROTATE_180)
                dst
            }
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 -> {
                val dst = Mat()
                Core.rotate(mat, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
                dst
            }
            else -> mat
        }

        Log.i("SFM_SIZE", "uriToMat: ${rotated.cols()}x${rotated.rows()} orientation=$orientation")
        return rotated
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
        /*if (req == CAMERA_PERMISSION_CODE && g.isNotEmpty() &&
            g[0] == PackageManager.PERMISSION_GRANTED) openCamera()*/
        if (req == CAMERA_PERMISSION_CODE && g.isNotEmpty() &&
            g[0] == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Camera permission granted. Press Open Camera again.", Toast.LENGTH_SHORT).show()
        }
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

    private fun saveSession() {
        try {
            if (savedImages.isEmpty()) {
                Toast.makeText(this, "No images to save!", Toast.LENGTH_SHORT).show()
                return
            }
            val dir = getExternalFilesDir("Sessions")
            dir?.mkdirs()
            val file = File(dir, "last_session.txt")
            file.writeText(savedImages.joinToString("\n") { it.toString() })
            Toast.makeText(this, "Session saved: ${savedImages.size} images", Toast.LENGTH_SHORT).show()
            savedImages.forEachIndexed { i, uri ->
                Log.i("SESSION", "save[$i] uri=$uri scheme=${uri.scheme} path=${uri.path}")
            }
            Log.i("SESSION", "Saved ${savedImages.size} URIs to ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e("SESSION", "Save failed", e)
            Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun loadSession() {
        try {
            val file = File(getExternalFilesDir("Sessions"), "last_session.txt")
            if (!file.exists()) {
                Toast.makeText(this, "No saved session found", Toast.LENGTH_SHORT).show()
                return
            }

            savedImages.clear()
            var loaded = 0
            var skipped = 0

            file.readLines()
                .filter { it.isNotBlank() }
                .forEach { line ->
                    try {
                        val uri = Uri.parse(line)

                        val canOpen = try {
                            contentResolver.openInputStream(uri)?.use { true } ?: false
                        } catch (e: Exception) {
                            false
                        }

                        if (canOpen) {
                            savedImages.add(uri)
                            loaded++
                            Log.i("SESSION", "Loaded URI: $uri")
                        } else {
                            Log.w("SESSION", "Skipping unreadable URI: $line")
                            skipped++
                        }
                    } catch (e: Exception) {
                        Log.w("SESSION", "Failed to parse URI: $line", e)
                        skipped++
                    }
                }

            val msg = if (skipped > 0)
                "Loaded $loaded images ($skipped unreadable)"
            else
                "Loaded $loaded images"

            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
            Log.i("SESSION", "Loaded $loaded URIs, skipped $skipped")
        } catch (e: Exception) {
            Log.e("SESSION", "Load failed", e)
            Toast.makeText(this, "Load failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun filterMatchesByMotionConsistency(
        matchSet: MatchSet,
        maxDeviationMultiplier: Double = 2.5
    ) {
        val (pts1, pts2) = matchSet.getMatchedPoints()

        if (pts1.size < 10 || pts1.size != pts2.size) return

        val dxs = pts1.indices.map { i -> pts2[i].x - pts1[i].x }
        val dys = pts1.indices.map { i -> pts2[i].y - pts1[i].y }

        fun median(values: List<Double>): Double {
            val sorted = values.sorted()
            return sorted[sorted.size / 2]
        }

        val medDx = median(dxs)
        val medDy = median(dys)

        val deviations = pts1.indices.map { i ->
            val ddx = dxs[i] - medDx
            val ddy = dys[i] - medDy
            Math.hypot(ddx, ddy)
        }

        val medDev = median(deviations)
        val threshold = maxOf(8.0, medDev * maxDeviationMultiplier)

        val kept1 = ArrayList<Point>()
        val kept2 = ArrayList<Point>()

        for (i in pts1.indices) {
            val ddx = dxs[i] - medDx
            val ddy = dys[i] - medDy
            val dev = Math.hypot(ddx, ddy)

            if (dev <= threshold) {
                kept1.add(pts1[i])
                kept2.add(pts2[i])
            }
        }

        Log.i(
            "SFM_MATCH_FILTER",
            "motion consistency: kept=${kept1.size}/${pts1.size}, " +
                    "medDx=${"%.1f".format(medDx)}, medDy=${"%.1f".format(medDy)}, " +
                    "threshold=${"%.1f".format(threshold)}"
        )

        matchSet.replaceMatches(kept1, kept2)
    }
    private fun filterMatchesByVerticalDisparity(
        matchSet: MatchSet,
        maxDyPx: Double = 8.0
    ) {
        val (pts1, pts2) = matchSet.getMatchedPoints()

        if (pts1.size != pts2.size) return

        val kept1 = ArrayList<Point>()
        val kept2 = ArrayList<Point>()

        for (i in pts1.indices) {
            val dy = Math.abs(pts2[i].y - pts1[i].y)

            if (dy <= maxDyPx) {
                kept1.add(pts1[i])
                kept2.add(pts2[i])
            }
        }

        Log.i(
            "SFM_MATCH_FILTER",
            "vertical disparity: kept=${kept1.size}/${pts1.size}, maxDy=$maxDyPx"
        )

        matchSet.replaceMatches(kept1, kept2)
    }

    private fun computeTrackReprojectionError(
        point3D: Point3,
        track: FeatureTrack,
        rotations: List<Mat>,
        translations: List<Mat>,
        K: Mat,
        D: Mat
    ): Double {
        val errors = mutableListOf<Double>()

        for (obs in track.observations) {
            val frameIdx = obs.frameIndex

            if (frameIdx !in rotations.indices || frameIdx !in translations.indices) {
                continue
            }

            val R = rotations[frameIdx]
            val t = translations[frameIdx]

            if (R.empty() || t.empty()) continue

            val rvec = Mat()
            val objectPoints = MatOfPoint3f(point3D)
            val projected = MatOfPoint2f()

            val distCoeffs = MatOfDouble()
            D.convertTo(distCoeffs, CvType.CV_64F)

            try {
                Calib3d.Rodrigues(R, rvec)

                Calib3d.projectPoints(
                    objectPoints,
                    rvec,
                    t,
                    K,
                    distCoeffs,
                    projected
                )

                val p = projected.toArray().firstOrNull() ?: continue

                val dx = p.x - obs.point2D.x
                val dy = p.y - obs.point2D.y
                val err = Math.hypot(dx, dy)

                if (err.isFinite()) {
                    errors.add(err)
                }
            } catch (e: Exception) {
                Log.w("SFM_TRACK_REPROJ", "Projection failed: ${e.message}")
            } finally {
                rvec.release()
                objectPoints.release()
                projected.release()
            }
        }

        if (errors.isEmpty()) return Double.MAX_VALUE

        return errors.average()
    }

}