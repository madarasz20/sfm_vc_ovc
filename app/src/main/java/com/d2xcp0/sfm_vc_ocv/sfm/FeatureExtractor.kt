package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.imgproc.Imgproc
import org.opencv.features2d.ORB

class FeatureExtractor {

    private val detector = ORB.create(
        //3000 //number of features
        6000,   // nfeatures
        1.2f,   // scaleFactor
        8,      // nlevels
        50,     // edgeThreshold "how much to ignore the sides of the image" (focus on the center part) pixelben
        0,      // firstLevel
        2,      // WTA_K
        ORB.HARRIS_SCORE,
        50,     // patchSize
        20      // fastThreshold    strongpoint detection (a lényegesebb pontokat pl sarok ilyenek)
    )

    fun compute(image: Mat): Pair<MatOfKeyPoint, Mat> {

        //Convert to grayscale for orb
        val gray = Mat()
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY)

        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()

        detector.detectAndCompute(
            gray,
            Mat(),
            keypoints,
            descriptors
        )

        return keypoints to descriptors
    }
}
