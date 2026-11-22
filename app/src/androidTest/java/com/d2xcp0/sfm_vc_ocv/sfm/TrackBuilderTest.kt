package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.KeyPoint
import org.opencv.core.MatOfKeyPoint

/**
 * U4, U5: a track-epites osszevonasi es utkozes-kezelesi logikaja.
 * Kepenkent 5 kulcspont; a parositasokat kezzel adjuk meg (kepindex, kulcspontindex).
 */
@RunWith(AndroidJUnit4::class)
class TrackBuilderTest {

    companion object {
        private const val TAG = "SFM_UNIT"
    }

    @Before
    fun loadOpenCV() {
        assertTrue("OpenCV not loaded", OpenCVLoader.initLocal())
    }

    private fun keypoints(n: Int = 5) =
        MatOfKeyPoint(*Array(n) { KeyPoint(10f * it, 10f * it, 1f) })

    private fun matches(vararg pairs: Pair<Int, Int>): MatchSet {
        val m = MatchSet(keypoints(), keypoints())
        for ((i, j) in pairs) m.addMatch(i, j)
        return m
    }

    private fun frames(track: FeatureTrack) = track.observations.map { it.frameIndex }

    // U4a: (0,1), (1,2), (0,2) parositas ugyanarra a pontra -> egyetlen, 3 kepes track
    @Test
    fun consistentTriangleGivesOneTrack() {
        val tb = TrackBuilder()
        tb.addPairMatches(0, 1, matches(0 to 1))
        tb.addPairMatches(1, 2, matches(1 to 2))
        tb.addPairMatches(0, 2, matches(0 to 2))

        val tracks = tb.getTracks()
        Log.i(TAG, "U4a track triangle: tracks=${tracks.size}, frames=${tracks.map { frames(it) }}")
        assertEquals(1, tracks.size)
        assertEquals(listOf(0, 1, 2), frames(tracks[0]))
    }

    // U4b: ket kulon track (0-1 es 2-3), majd egy (1,2) parositas osszekoti oket.
    // Nincs kozos kepuk, ezert egyetlen, 4 kepes track-ke kell osszeolvadniuk.
    @Test
    fun mergesTracksWithoutCommonFrame() {
        val tb = TrackBuilder()
        tb.addPairMatches(0, 1, matches(0 to 1))
        tb.addPairMatches(2, 3, matches(2 to 3))
        assertEquals(2, tb.getTracks().size)

        tb.addPairMatches(1, 2, matches(1 to 2))

        val tracks = tb.getTracks()
        Log.i(TAG, "U4b track merge: tracks=${tracks.size}, frames=${tracks.map { frames(it) }}")
        assertEquals(1, tracks.size)
        assertEquals(listOf(0, 1, 2, 3), frames(tracks[0]))
    }

    // U5: ket track, amelyek ugyanazon a kepen (0) kulonbozo kulcspontot tartalmaznak.
    // Egy oket osszekoto parositas ellentmondas, ezert nem szabad osszeolvasztani.
    @Test
    fun conflictingTracksAreNotMerged() {
        val tb = TrackBuilder()
        tb.addPairMatches(0, 1, matches(0 to 1))   // A track: kep0/kp0 - kep1/kp1
        tb.addPairMatches(0, 2, matches(3 to 2))   // B track: kep0/kp3 - kep2/kp2
        tb.addPairMatches(1, 2, matches(1 to 2))   // A es B osszekotese -> utkozes a 0. kepen

        val tracks = tb.getTracks()
        Log.i(TAG, "U5 track conflict: tracks=${tracks.size}, frames=${tracks.map { frames(it) }}")
        assertEquals(2, tracks.size)
        for (t in tracks) {
            val f = frames(t)
            assertEquals("a track contains the same frame twice", f.size, f.distinct().size)
        }
    }
}
