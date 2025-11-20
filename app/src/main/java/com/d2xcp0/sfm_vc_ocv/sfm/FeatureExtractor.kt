package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.features2d.Feature2D
import org.opencv.features2d.ORB

class FeatureExtractor {

    private val detector: Feature2D = ORB.create(2000)   // Safe for Android

    fun compute(image: Mat): Pair<MatOfKeyPoint, Mat> {

        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()

        detector.detectAndCompute(
            image,
            Mat(),          // no mask
            keypoints,
            descriptors
        )

        return keypoints to descriptors
    }
}
