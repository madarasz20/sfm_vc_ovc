package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.imgproc.Imgproc
import org.opencv.features2d.ORB

class FeatureExtractor {

    // ORB is perfect for mobile SfM
    private val detector = ORB.create(
        2000 // number of features
    )

    fun compute(image: Mat): Pair<MatOfKeyPoint, Mat> {

        // Convert to grayscale (ORB requires single-channel)
        val gray = Mat()
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY)

        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()

        detector.detectAndCompute(
            gray,
            Mat(),      // no mask
            keypoints,
            descriptors
        )

        return keypoints to descriptors
    }
}
