package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.calib3d.Calib3d
import org.opencv.core.*

class PoseEstimator(private val K: Mat) {

    fun estimatePose(matchSet: MatchSet): Pair<Mat, Mat> {

        val (pts1, pts2) = matchSet.getMatchedPoints()

        val mat1 = MatOfPoint2f(*pts1.toTypedArray())
        val mat2 = MatOfPoint2f(*pts2.toTypedArray())

        val focal = K.get(0, 0)[0]
        val cx = K.get(0, 2)[0]
        val cy = K.get(1, 2)[0]
        val pp = Point(cx, cy)

        val E = Calib3d.findEssentialMat(mat1, mat2, focal, pp)

        val R = Mat()
        val t = Mat()

        Calib3d.recoverPose(E, mat1, mat2, R, t, focal, pp)

        return R to t
    }
}
