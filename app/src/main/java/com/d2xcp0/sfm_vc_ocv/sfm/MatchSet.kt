package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.Point
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f

data class IndexedMatch(
    val idx1: Int,
    val idx2: Int,
    val pt1: Point,
    val pt2: Point
)
class MatchSet(
    keypoints1: MatOfKeyPoint,
    keypoints2: MatOfKeyPoint
) {

    private val kp1Array = keypoints1.toArray()
    private val kp2Array = keypoints2.toArray()

    private val pts1 = mutableListOf<Point>()
    private val pts2 = mutableListOf<Point>()

    private val indexedMatches = mutableListOf<IndexedMatch>()

    fun addMatch(i: Int, j: Int) {
        if (i < 0 || i >= kp1Array.size || j < 0 || j >= kp2Array.size) {
            android.util.Log.e(
                "MatchSet",
                "Invalid addMatch indices: i=$i/${kp1Array.size}, j=$j/${kp2Array.size}"
            )
            return
        }

        indexedMatches.add(
            IndexedMatch(
                idx1 = i,
                idx2 = j,
                pt1 = kp1Array[i].pt,
                pt2 = kp2Array[j].pt
            )
        )
    }

    fun replaceMatches(new1: List<Point>, new2: List<Point>) {
        /*pts1.clear()
        pts2.clear()
        pts1.addAll(new1)
        pts2.addAll(new2)*/

        indexedMatches.clear()

        val n = minOf(new1.size, new2.size)

        for (i in 0 until n) {
            // Index is unknown after point-only replacement.
            // Use -1 so track builder can ignore these if needed.
            indexedMatches.add(
                IndexedMatch(
                    idx1 = -1,
                    idx2 = -1,
                    pt1 = new1[i],
                    pt2 = new2[i]
                )
            )
        }

    }

    /*fun getMatchedPoints(): Pair<List<Point>, List<Point>> {

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



    val size: Int get() = pts1.size*/

    fun replaceIndexedMatches(newMatches: List<IndexedMatch>) {
        indexedMatches.clear()
        indexedMatches.addAll(newMatches)
    }

    fun getIndexedMatches(): List<IndexedMatch> {
        return indexedMatches.toList()
    }

    fun getMatchedPoints(): Pair<List<Point>, List<Point>> {
        return indexedMatches.map { it.pt1 } to indexedMatches.map { it.pt2 }
    }

    fun getMatchedPointMats(): Pair<MatOfPoint2f, MatOfPoint2f> {
        val (pts1, pts2) = getMatchedPoints()
        return Pair(
            MatOfPoint2f(*pts1.toTypedArray()),
            MatOfPoint2f(*pts2.toTypedArray())
        )
    }

    fun toListOfPairs(): List<Pair<Point, Point>> {
        return indexedMatches.map { it.pt1 to it.pt2 }
    }

    val size: Int get() = indexedMatches.size


}
