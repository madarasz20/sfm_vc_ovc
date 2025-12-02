package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.*
import org.opencv.features2d.FlannBasedMatcher

class AnchorMatcher {

    private val matcher = FlannBasedMatcher()

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

        // --- SAFE ZERO CHECK ---
        if (anchor3D.isEmpty() || anchorDesc.empty() || descriptors.empty()) {
            return Pair(emptyList(), emptyList())
        }

        // FLANN requires CV_32F
        val d1 = Mat()
        val d2 = Mat()
        anchorDesc.convertTo(d1, CvType.CV_32F)
        descriptors.convertTo(d2, CvType.CV_32F)

        val kp2 = keypoints.toArray()
        if (kp2.isEmpty()) {
            return Pair(emptyList(), emptyList())
        }

        val knnMatches = ArrayList<MatOfDMatch>()
        matcher.knnMatch(d1, d2, knnMatches, 2)

        val out3D = ArrayList<Point3>()
        val out2D = ArrayList<Point>()

        // SAFETY LIMITS
        val max3D = anchor3D.size
        val maxKP = kp2.size
        val maxDesc1 = d1.rows()
        val maxDesc2 = d2.rows()

        for (m in knnMatches) {
            val arr = m.toArray()
            if (arr.size < 2) continue

            val best = arr[0]
            val second = arr[1]

            // Lowe ratio test
            if (best.distance >= 0.7f * second.distance) continue

            val q = best.queryIdx   // index into anchorDesc and anchor3D
            val t = best.trainIdx   // index into descriptors and kp2

            // --- FULL SAFETY CHECKS ---
            if (q !in 0 until max3D) continue
            if (q !in 0 until maxDesc1) continue
            if (t !in 0 until maxKP) continue
            if (t !in 0 until maxDesc2) continue

            // Valid match
            out3D.add(anchor3D[q])
            out2D.add(kp2[t].pt)
        }

        return Pair(out3D, out2D)
    }
}
