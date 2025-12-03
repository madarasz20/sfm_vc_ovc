package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.core.*
import org.opencv.features2d.BFMatcher
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfDMatch

class FeatureMatcher {

    companion object {
        private const val TAG = "FeatureMatcher"
    }
    // featurematcher for ORB
    private val matcher = BFMatcher.create(Core.NORM_HAMMING, true)
    // crossCheck = true

    fun match(
        desc1: Mat,
        desc2: Mat,
        kp1: MatOfKeyPoint,
        kp2: MatOfKeyPoint
    ): MatchSet {

        val matchSet = MatchSet(kp1, kp2)

        if (desc1.empty() || desc2.empty()) {
            Log.w(TAG, "Empty descriptors, skipping match.")
            return matchSet
        }

        val matches = MatOfDMatch()
        matcher.match(desc1, desc2, matches)

        val goodMatches = mutableListOf<DMatch>()
        for (m in matches.toArray()) {
            if (m.distance < 40) {  // threshold for ORB
                goodMatches.add(m)
            }
        }

        Log.i(TAG, "Matches: total=${matches.toArray().size}, good=${goodMatches.size}")

        for (gm in goodMatches) {
            matchSet.addMatch(gm.queryIdx, gm.trainIdx)
        }

        return matchSet
    }
}
