package com.d2xcp0.sfm_vc_ocv.sfm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.max

class CameraCalibrator(private val context: Context) {

    private val TAG = "CALIB"

    private val boardSize = Size(9.0, 6.0)
    private val squareSize = 28.0  // ← CONFIRM this against your actual board

    // FIX 1: Must match resizeForCalibration() in MainActivity exactly.
    // K is only valid for the resolution it was computed at.
    companion object {
        const val TARGET_SIZE = 1200.0
    }

    fun loadCalibrationImages(): List<Mat> {
        val mats = mutableListOf<Mat>()
        val assetManager = context.assets
        val files = assetManager.list("calibration") ?: emptyArray()

        Log.i(TAG, "Found ${files.size} calibration images")

        for (file in files) {
            try {
                val bitmap: Bitmap?

                assetManager.open("calibration/$file").use { input ->
                    val options = BitmapFactory.Options().apply {
                        inPreferredConfig = Bitmap.Config.RGB_565
                    }
                    bitmap = BitmapFactory.decodeStream(input, null, options)
                }

                if (bitmap == null) {
                    Log.e(TAG, "Failed decoding bitmap: $file")
                    continue
                }

                val rgba = Mat()
                Utils.bitmapToMat(bitmap, rgba)
                bitmap.recycle()

                val gray = Mat()
                Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
                rgba.release()

                // FIX 1: Resize calibration images to the same resolution
                // used by the SfM pipeline. K is only valid at the resolution
                // it was computed at — this was the root cause of wrong fx/fy/cx/cy.
                val resized = resizeToTarget(gray)

                mats.add(resized)
                Log.i(TAG, "Loaded $file → ${resized.width()}x${resized.height()}")

            } catch (e: Exception) {
                Log.e(TAG, "Failed loading $file: ${e.message}", e)
            }
        }

        return mats
    }

    fun calibrate(images: List<Mat>): Boolean {

        if (images.isEmpty()) {
            Log.e(TAG, "No images provided to calibration")
            return false
        }

        val objectPoints = mutableListOf<Mat>()
        val imagePoints  = mutableListOf<Mat>()

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
            Log.i(TAG, "Detecting corners in image $index (${img.width()}x${img.height()})...")

            val corners = MatOfPoint2f()

            // findChessboardCornersSB does its own sub-pixel refinement internally.
            val found = Calib3d.findChessboardCornersSB(img, boardSize, corners)

            if (found) {
                foundCount++

                // FIX 2: Removed redundant cornerSubPix call.
                // findChessboardCornersSB already performs sub-pixel refinement.
                // Running it again can slightly degrade accuracy.

                objectPoints.add(objMat.clone())
                imagePoints.add(corners)
                Log.i(TAG, "✔ Corners FOUND in image $index")
            } else {
                Log.w(TAG, "✘ Corners NOT found in image $index")
            }
        }

        Log.i(TAG, "Found corners in $foundCount / ${images.size} images")

        if (foundCount < 5) {
            Log.e(TAG, "Too few valid images for calibration ($foundCount)")
            return false
        }

        val K    = Mat.eye(3, 3, CvType.CV_64F)

        // FIX 3: Use 5 distortion coefficients instead of 8.
        // The 8-coefficient rational model can overfit with ~20 images,
        // producing a distortion matrix that hurts rather than helps.
        // Standard 5-coeff (k1, k2, p1, p2, k3) is stable and sufficient
        // for a phone lens at this image count.
        val dist = Mat.zeros(5, 1, CvType.CV_64F)

        val rvecs = ArrayList<Mat>()
        val tvecs = ArrayList<Mat>()

        val rms = Calib3d.calibrateCamera(
            objectPoints,
            imagePoints,
            images[0].size(),  // now correctly reflects TARGET_SIZE resolution
            K,
            dist,
            rvecs,
            tvecs
        )

        Log.i(TAG, "Calibration RMS error: $rms")
        Log.i(TAG, "K =\n${K.dump()}")
        Log.i(TAG, "D =\n${dist.dump()}")

        // RMS above ~1.5px usually means something is wrong
        if (rms > 1.5) {
            Log.w(TAG, "High RMS error ($rms) — check board images for blur or bad angles")
        }

        CalibrationStorage.save(context, K, dist)
        return true
    }

    // Resize to TARGET_SIZE on the longest dimension, preserving aspect ratio.
    // Identical logic to resizeForCalibration() in MainActivity.
    private fun resizeToTarget(src: Mat): Mat {
        val w = src.width().toDouble()
        val h = src.height().toDouble()
        val scale = TARGET_SIZE / max(w, h)

        if (scale >= 1.0) return src.clone()

        val dst = Mat()
        Imgproc.resize(src, dst, Size(w * scale, h * scale))
        return dst
    }
}