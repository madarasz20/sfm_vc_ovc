package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.features2d.ORB
import org.opencv.features2d.SIFT
import android.util.Log

//import org.opencv.imgproc.Imgproc.Canny

class FeatureExtractor {

    //private val c_detector = Canny.create().
    private val detector = ORB.create(
        //3000 //number of features
        6000,   // nfeatures
        1.35f,   // scaleFactor 1.2
        8,      // nlevels 8
        50,     // edgeThreshold "how much to ignore the sides of the image" (focus on the center part) pixelben, kisebb targynal lehet pl.35-30 is akar meg kell nezni
        0,      // firstLevel
        2,      // WTA_K
        ORB.HARRIS_SCORE,
        50,     // patchSize megegyezik az edge tresholddal
        15      // fastThreshold    strongpoint detection (a lényegesebb pontokat pl sarok ilyenek alacsony texturaju kornyezetben pl 10-15 is lehet)
    )
    /*SIFT.create(
        800,        // nFeatures: increase to detect more points
        3,          // nOctaveLayers
        0.02,       // contrastThreshold: LOWER = more features
        10.0,       // edgeThreshold
        1.6         // sigma
    )*/

    fun compute(image: Mat): Pair<MatOfKeyPoint, Mat> {

        //Convert to grayscale for orb
        val gray = Mat()
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY)
        Log.i("SFM_INPUT", "FeatureExtractor input image = ${image.cols()}x${image.rows()}, type=${image.type()}, channels=${image.channels()}")

        val mask = Mat.zeros(gray.size(), CvType.CV_8UC1)
        val w = gray.width()
        val h = gray.height()
        val roi = Rect(
            (w * 0.05).toInt(),
            (h * 0.05).toInt(),
            (w * 0.90).toInt(),
            (h * 0.90).toInt()
        )
        mask.submat(roi).setTo(Scalar(255.0))
        Log.i("SFM_INPUT", "FeatureExtractor ROI = x=${roi.x}, y=${roi.y}, w=${roi.width}, h=${roi.height}")

        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()

        detector.detectAndCompute(
            gray,
             mask,
            keypoints,
            descriptors
        )

        Log.i("SFM_MATCH", "FeatureExtractor keypoints = ${keypoints.toArray().size}")
        Log.i("SFM_MATCH", "FeatureExtractor descriptors = ${descriptors.rows()}x${descriptors.cols()}, type=${descriptors.type()}")



        return keypoints to descriptors
    }
}
