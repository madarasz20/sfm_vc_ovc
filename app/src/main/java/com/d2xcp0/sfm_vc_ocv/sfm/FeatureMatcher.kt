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
    // crossCheck = true ha nincs knn

    private val  RATIO = 0.80f      //lowe
    private val MIN_GAP = 5.0f      //best match has to be at least X bits better
    private val MAX_DISTANCE = 65.0f    //Highest acceptable Hamming distance


    fun match(
        desc1: Mat,
        desc2: Mat,
        kp1: MatOfKeyPoint,
        kp2: MatOfKeyPoint
    ): MatchSet {

        val matchSet = MatchSet(kp1, kp2)
        Log.i("SFM_MATCH", "kp1 = ${kp1.toArray().size}")
        Log.i("SFM_MATCH", "desc1 = ${desc1.rows()}x${desc1.cols()}, type=${desc1.type()}")
        //, dump=${desc1.dump()}
        Log.i("SFM_MATCH", "kp2 = ${kp2.toArray().size}")
        Log.i("SFM_MATCH", "desc2 = ${desc2.rows()}x${desc2.cols()}, type=${desc2.type()}")
        //, dump=${desc2.dump()}

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

                /*if (best.distance < 0.45f * second.distance) {
                    goodMatches.add(best)
                }*/
                if (
                    best.distance <= MAX_DISTANCE &&
                    best.distance < RATIO * second.distance &&
                    second.distance - best.distance >= MIN_GAP
                ) {
                    goodMatches.add(best)
                }
            }
        }
        Log.i("SFM_MATCH", "raw knn matches = ${knnMatches.size}")
        Log.i("SFM_MATCH", "good matches after ratio = ${goodMatches.size}")


        Log.i(TAG, "Raw matches=${knnMatches.size}, good=${goodMatches.size}")
        val rawMatches = MatOfDMatch()
        matcher.match(desc1, desc2, rawMatches)

        /*val goodMatches = rawMatches.toArray()
            .filter { it.distance < 40.0 }
            .sortedBy { it.distance }
            .take(500)

        Log.i("SFM_MATCH", "crossCheck matches=${goodMatches.size}")*/

        for (gm in goodMatches) {
            matchSet.addMatch(gm.queryIdx, gm.trainIdx)
        }

        /*val forwardKnn = ArrayList<MatOfDMatch>()
        val reverseKnn = ArrayList<MatOfDMatch>()

        matcher.knnMatch(desc1, desc2, forwardKnn, 2)
        matcher.knnMatch(desc2, desc1, reverseKnn, 2)

        fun ratioFilter(knn: ArrayList<MatOfDMatch>): List<DMatch> {
            val result = mutableListOf<DMatch>()

            for (matMatch in knn) {
                val arr = matMatch.toArray()
                if (arr.size < 2) continue

                val best = arr[0]
                val second = arr[1]

                if (best.distance < 0.65f * second.distance) {
                    result.add(best)
                }
            }

            return result
        }

        val forwardGood = ratioFilter(forwardKnn)
        val reverseGood = ratioFilter(reverseKnn)

        val reversePairs = reverseGood
            .map { rev ->
                rev.queryIdx to rev.trainIdx
            }
            .toSet()

        val mutualGood = forwardGood
            .filter { fwd ->
                val reversePair = fwd.trainIdx to fwd.queryIdx
                reversePair in reversePairs
            }
            .sortedBy { it.distance }
            .take(500)

        Log.i("SFM_MATCH", "raw forward knn=${forwardKnn.size}")
        Log.i("SFM_MATCH", "forward ratio=${forwardGood.size}")
        Log.i("SFM_MATCH", "reverse ratio=${reverseGood.size}")
        Log.i("SFM_MATCH", "mutual ratio=${mutualGood.size}")

        for (gm in mutualGood) {
            matchSet.addMatch(gm.queryIdx, gm.trainIdx)
        }*/

        return matchSet
    }
}
