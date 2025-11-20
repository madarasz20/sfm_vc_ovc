package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.KeyPoint
import org.opencv.core.Point
import org.opencv.core.MatOfKeyPoint

class MatchSet(
    val keypoints1: MatOfKeyPoint,
    val keypoints2: MatOfKeyPoint
) {

    private val pairs = mutableListOf<Pair<Int, Int>>()

    fun addMatch(i: Int, j: Int) {
        pairs.add(i to j)
    }

    // Returns (points1, points2) as lists of 2D point coordinates
    fun getMatchedPoints(): Pair<List<Point>, List<Point>> {

        val pts1 = mutableListOf<Point>()
        val pts2 = mutableListOf<Point>()

        val kp1Array = keypoints1.toArray()
        val kp2Array = keypoints2.toArray()

        for ((i, j) in pairs) {
            pts1.add(kp1Array[i].pt)
            pts2.add(kp2Array[j].pt)
        }

        return pts1 to pts2
    }
}
