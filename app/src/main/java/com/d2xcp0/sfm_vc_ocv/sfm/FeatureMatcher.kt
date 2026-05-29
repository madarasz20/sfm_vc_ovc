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
    private val matcher = BFMatcher.create(Core.NORM_HAMMING, false)
    // crossCheck = true

    fun match(
        desc1: Mat,
        desc2: Mat,
        kp1: MatOfKeyPoint,
        kp2: MatOfKeyPoint
    ): MatchSet {

        val matchSet = MatchSet(kp1, kp2)
        Log.i("SFM_MATCH", "kp1 = ${kp1.toArray().size}, kp2 = ${kp2.toArray().size}")
        Log.i("SFM_MATCH", "desc1 = ${desc1.rows()}x${desc1.cols()}, type=${desc1.type()}")
        Log.i("SFM_MATCH", "desc2 = ${desc2.rows()}x${desc2.cols()}, type=${desc2.type()}")

        if (desc1.empty() || desc2.empty()) {
            Log.w(TAG, "Empty descriptors, skipping match.")
            return matchSet
        }

        /*val matches = MatOfDMatch()
        matcher.match(desc1, desc2, matches)

        val goodMatches = mutableListOf<DMatch>()
        for (m in matches.toArray()) {
            if (m.distance < 40) {  // threshold for ORB
                goodMatches.add(m)
            }
        }*/

        val knnMatches = ArrayList<MatOfDMatch>()

        matcher.knnMatch(desc1, desc2, knnMatches, 2)

        val goodMatches = mutableListOf<DMatch>()

        for (matMatch in knnMatches) {

            val matches = matMatch.toArray()

            if (matches.size >= 2) {

                val best = matches[0]
                val second = matches[1]
                    //TODO 0.7-el is nezd meg
                if (best.distance < 0.70f * second.distance) {
                    goodMatches.add(best)
                }
            }
        }
        Log.i("SFM_MATCH", "raw knn matches = ${knnMatches.size}")
        Log.i("SFM_MATCH", "good matches after ratio = ${goodMatches.size}")


        Log.i(TAG, "Raw matches=${knnMatches.size}, good=${goodMatches.size}")


        for (gm in goodMatches) {
            matchSet.addMatch(gm.queryIdx, gm.trainIdx)
        }

        return matchSet
    }
}
