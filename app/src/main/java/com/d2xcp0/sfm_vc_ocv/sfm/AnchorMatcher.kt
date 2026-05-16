package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.*
import org.opencv.features2d.FlannBasedMatcher
import org.opencv.features2d.BFMatcher

class AnchorMatcher {

    private val matcher = BFMatcher.create(Core.NORM_HAMMING, false)

    /**
     * anchor3D: List<Point3>      – 3D anchor points
     * anchorDesc: Mat             – descriptors for anchor3D
     * keypoints: MatOfKeyPoint    – new frame keypoints
     * descriptors: Mat            – descriptors for new frame
     */
    fun match3DTo2D(
        anchor3D: List<Point3>,
        anchorDesc: Mat,
        keypoints: MatOfKeyPoint,
        descriptors: Mat
    ): Pair<List<Point3>, List<Point>> {

        // zero check
        if (anchor3D.isEmpty() || anchorDesc.empty() || descriptors.empty()) {
            return Pair(emptyList(), emptyList())
        }

        //val d1 = Mat()
        //val d2 = Mat()
        //anchorDesc.convertTo(d1, CvType.CV_32F)
        //descriptors.convertTo(d2, CvType.CV_32F)

        val kp2 = keypoints.toArray()
        if (kp2.isEmpty()) {
            return Pair(emptyList(), emptyList())
        }

        val knnMatches = ArrayList<MatOfDMatch>()
        matcher.knnMatch(anchorDesc, descriptors, knnMatches, 2)

        val out3D = ArrayList<Point3>()
        val out2D = ArrayList<Point>()

        //limits
        val max3D = anchor3D.size
        val maxKP = kp2.size
        //val maxDesc1 = d1.rows()
        //val maxDesc2 = d2.rows()

        for (m in knnMatches) {
            val arr = m.toArray()
            if (arr.size < 2) continue

            val best = arr[0]
            val second = arr[1]

            //Lowe ratio test
            if (best.distance >= 0.75f * second.distance) continue

            val q = best.queryIdx   //index into anchorDesc and anchor3D
            val t = best.trainIdx   //index into descriptors and kp2

            //full check
            if(q !in anchor3D.indices) continue
            if(t !in kp2.indices) continue

            // Valid match
            out3D.add(anchor3D[q])
            out2D.add(kp2[t].pt)
        }

        return Pair(out3D, out2D)
    }
}
