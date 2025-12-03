package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.Point
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f

class MatchSet(
    keypoints1: MatOfKeyPoint,
    keypoints2: MatOfKeyPoint
) {

    // Convert once, not every time addMatch() is called
    private val kp1Array = keypoints1.toArray()
    private val kp2Array = keypoints2.toArray()

    private val pts1 = mutableListOf<Point>()
    private val pts2 = mutableListOf<Point>()

    fun addMatch(i: Int, j: Int) {
        pts1.add(kp1Array[i].pt)
        pts2.add(kp2Array[j].pt)
    }

    fun replaceMatches(new1: List<Point>, new2: List<Point>) {
        pts1.clear()
        pts2.clear()
        pts1.addAll(new1)
        pts2.addAll(new2)
    }

    fun getMatchedPoints(): Pair<List<Point>, List<Point>> {

        return pts1 to pts2
    }

    fun getMatchedPointMats(): Pair<MatOfPoint2f, MatOfPoint2f> {
        return Pair(
            MatOfPoint2f(*pts1.toTypedArray()),
            MatOfPoint2f(*pts2.toTypedArray())
        )
    }

    fun toListOfPairs(): List<Pair<Point, Point>> {
        val (a, b) = getMatchedPoints()
        val out = ArrayList<Pair<Point,Point>>()
        val n = minOf(a.size, b.size)
        for (i in 0 until n) out.add(Pair(a[i], b[i]))
        return out
    }



    val size: Int get() = pts1.size
}
