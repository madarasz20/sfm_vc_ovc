package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

class CameraCalibrator(
    private val boardSize: Size,       // e.g. Size(9.0, 6.0)
    private val squareSize: Double     // e.g. 0.024
) {
    private val objectPoints = mutableListOf<Mat>()  // MUST be Mat
    private val imagePoints  = mutableListOf<Mat>()  // MUST be Mat

    private var K  = Mat()
    private var dist = Mat()

    fun calibrate(chessboardImages: List<Mat>) {

        // Build the 3D template pattern
        val objCorners = MatOfPoint3f()
        val objList = ArrayList<Point3>()
        for (j in 0 until boardSize.height.toInt()) {
            for (i in 0 until boardSize.width.toInt()) {
                objList.add(Point3(i * squareSize, j * squareSize, 0.0))
            }
        }
        objCorners.fromList(objList)

        for (img in chessboardImages) {

            val gray = Mat()
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY)

            val imgCorners = MatOfPoint2f()
            val found = Calib3d.findChessboardCorners(gray, boardSize, imgCorners)

            if (found) {
                // MUST convert to Mat
                objectPoints.add(objCorners)
                imagePoints.add(imgCorners)
            }
        }

        val imageSize = chessboardImages.first().size()

        // Prepare output matrices
        K = Mat.eye(3, 3, CvType.CV_64F)
        dist = Mat.zeros(8, 1, CvType.CV_64F)

        Calib3d.calibrateCamera(
            objectPoints,
            imagePoints,
            imageSize,
            K,
            dist,
            mutableListOf<Mat>(),
            mutableListOf<Mat>()
        )
    }

    fun getIntrinsicMatrix(): Mat = K
    fun getDistortion(): Mat = dist
}