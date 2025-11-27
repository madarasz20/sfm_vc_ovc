package com.d2xcp0.sfm_vc_ocv.sfm

import android.content.Context
import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.android.Utils
import android.graphics.BitmapFactory
import kotlin.math.max


class CameraCalibrator(private val context: Context) {

    private val TAG = "CALIB"

    // Board: 9x6 INNER corners (your chessboard)
    private val boardSize = Size(9.0, 6.0)
    private val squareSize = 25.0   // mm or any unit

    // Load images from assets/calibration/
    fun loadCalibrationImages(): List<Mat> {
        val mats = mutableListOf<Mat>()
        val assetManager = context.assets

        val files = assetManager.list("calibration") ?: emptyArray()

        Log.i(TAG, "Found ${files.size} calibration images")

        for (file in files) {
            try {
                val input = assetManager.open("calibration/$file")
                val bitmap = BitmapFactory.decodeStream(input)

                val mat = Mat()
                Utils.bitmapToMat(bitmap, mat)    // RGBA
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY)

                // 🔥 DOWNSCALE LARGE IMAGES (CRUCIAL)
                val maxDim = 1200.0
                val scale = maxDim / max(mat.width().toDouble(), mat.height().toDouble())

                if (scale < 1.0) {
                    Imgproc.resize(mat, mat, Size(mat.width() * scale, mat.height() * scale))
                    Log.i(TAG, "Downscaled $file → ${mat.width()}x${mat.height()}")
                }

                mats.add(mat)
                Log.i(TAG, "Loaded $file")

            } catch (e: Exception) {
                Log.e(TAG, "Failed loading $file : ${e.message}")
            }
        }

        return mats
    }


    // Perform calibration
    fun calibrate(images: List<Mat>): Boolean {

        if (images.isEmpty()) {
            Log.e(TAG, "No images provided to calibration")
            return false
        }

        val objectPoints = mutableListOf<Mat>()
        val imagePoints  = mutableListOf<Mat>()

        // Prepare 3D chessboard coordinates
        val objList = ArrayList<Point3>()
        for (y in 0 until boardSize.height.toInt()) {
            for (x in 0 until boardSize.width.toInt()) {
                objList.add(Point3(x * squareSize, y * squareSize, 0.0))
            }
        }
        val objMat = MatOfPoint3f()
        objMat.fromList(objList)

        var foundCount = 0

        for ((index, img) in images.withIndex()) {

            Log.i(TAG, "Detecting corners in image $index...")

            val corners = MatOfPoint2f()
            val found = Calib3d.findChessboardCorners(
                img, boardSize, corners,
                Calib3d.CALIB_CB_ADAPTIVE_THRESH + Calib3d.CALIB_CB_NORMALIZE_IMAGE
            )

            if (found) {
                foundCount++

                Imgproc.cornerSubPix(
                    img, corners,
                    Size(11.0, 11.0), Size(-1.0, -1.0),
                    TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.1)
                )

                objectPoints.add(objMat)
                imagePoints.add(corners)

                Log.i(TAG, "✔ Corners FOUND in image $index")

            } else {
                Log.w(TAG, "✘ Corners NOT found in image $index")
            }
        }

        Log.i(TAG, "Found corners in $foundCount / ${images.size} images")

        if (foundCount < 5) {
            Log.e(TAG, "Too few valid images for calibration")
            return false
        }

        val K = Mat.eye(3, 3, CvType.CV_64F)
        val dist = Mat.zeros(8, 1, CvType.CV_64F)

        val rvecs = ArrayList<Mat>()
        val tvecs = ArrayList<Mat>()

        val rms = Calib3d.calibrateCamera(
            objectPoints,
            imagePoints,
            images[0].size(),
            K,
            dist,
            rvecs,
            tvecs
        )

        Log.i(TAG, "Calibration RMS error: $rms")
        Log.i(TAG, "Camera Matrix:\n$K")
        Log.i(TAG, "Distortion Coeffs:\n$dist")

        // Save to SharedPreferences
        saveCalibration(K, dist)

        return true
    }

    private fun saveCalibration(K: Mat, dist: Mat) {
        val prefs = context.getSharedPreferences("calib", Context.MODE_PRIVATE)
        val editor = prefs.edit()

        val Karr = DoubleArray(9)
        K.get(0, 0, Karr)

        val Darr = DoubleArray(dist.total().toInt())
        dist.get(0, 0, Darr)

        editor.putString("K", Karr.joinToString(","))
        editor.putString("D", Darr.joinToString(","))
        editor.apply()

        Log.i(TAG, "Calibration saved.")
    }
}
