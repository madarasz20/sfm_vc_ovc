package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.*
import org.opencv.features2d.BFMatcher
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint

class FeatureMatcher {

    // ORB works with Hamming distance
    private val matcher = BFMatcher.create(BFMatcher.BRUTEFORCE_HAMMING, true)

    fun match(
        desc1: Mat,
        desc2: Mat,
        kp1: MatOfKeyPoint,
        kp2: MatOfKeyPoint
    ): MatchSet {

        val matches = MatOfDMatch()
        matcher.match(desc1, desc2, matches)

        val matchArray = matches.toArray()

        // Filter good matches
        val minDist = matchArray.minOf { it.distance.toDouble() }
        val goodMatches = matchArray.filter { it.distance < minDist * 3.0 }

        val matchSet = MatchSet(kp1, kp2)

        for (m in goodMatches) {
            matchSet.addMatch(m.queryIdx, m.trainIdx)
        }

        return matchSet
    }
}
