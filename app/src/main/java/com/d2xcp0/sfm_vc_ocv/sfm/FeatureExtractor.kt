package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.features2d.Feature2D
import org.opencv.features2d.ORB
import org.opencv.features2d.SIFT
import android.util.Log
import kotlin.math.sqrt

//import org.opencv.imgproc.Imgproc.Canny

enum class FeatureType {
    ORB,        // binaris, Hamming tavolsag
    SIFT,       // float, L2 tavolsag
    ROOT_SIFT   // SIFT, L1 normalizalas + gyokvonas -> L2 ~ Hellinger kernel, jobb parositas
}

class FeatureExtractor(private val type: FeatureType = FeatureType.ROOT_SIFT) {

    //private val c_detector = Canny.create().
    private val detector: Feature2D = when (type) {
        FeatureType.ORB -> ORB.create(
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
        FeatureType.SIFT, FeatureType.ROOT_SIFT -> SIFT.create(
            4000,   // nFeatures: a legerosebb N pont marad
            3,      // nOctaveLayers
            0.02,   // contrastThreshold: alacsonyabb = tobb pont a gyengen texturalt feluleteken (alap 0.04)
            10.0,   // edgeThreshold
            1.6     // sigma
        )
    }

    fun compute(image: Mat): Pair<MatOfKeyPoint, Mat> {

        //Convert to grayscale for feature detection
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

        if (type == FeatureType.ROOT_SIFT) toRootSift(descriptors)

        Log.i("SFM_MATCH", "FeatureExtractor [$type] keypoints = ${keypoints.toArray().size}")
        Log.i("SFM_MATCH", "FeatureExtractor descriptors = ${descriptors.rows()}x${descriptors.cols()}, type=${descriptors.type()}")



        return keypoints to descriptors
    }

    // RootSIFT (Arandjelovic & Zisserman, 2012): soronkent L1 normalizalas, majd elemenkenti gyok.
    // A SIFT ertekek nemnegativak, igy az L1 norma egyszeruen az osszeg.
    private fun toRootSift(desc: Mat) {
        if (desc.empty() || desc.type() != CvType.CV_32F) return

        val rows = desc.rows()
        val cols = desc.cols()
        val data = FloatArray(rows * cols)
        desc.get(0, 0, data)

        for (r in 0 until rows) {
            val o = r * cols
            var sum = 0f
            for (c in 0 until cols) sum += data[o + c]
            if (sum <= 1e-7f) continue
            for (c in 0 until cols) data[o + c] = sqrt(data[o + c] / sum)
        }

        desc.put(0, 0, data)
    }
}
