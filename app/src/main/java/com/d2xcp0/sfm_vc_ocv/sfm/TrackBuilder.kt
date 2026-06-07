package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log

class TrackBuilder {

    companion object {
        private const val TAG = "SFM_TRACKS"
    }

    private val tracks = mutableListOf<FeatureTrack>()

    // Maps one detected keypoint in one frame to a track id.
    private val obsToTrack = mutableMapOf<Pair<Int, Int>, Int>()

    fun addPairMatches(
        frameLeft: Int,
        frameRight: Int,
        matches: MatchSet
    ) {
        val allIndexed = matches.getIndexedMatches()
        val indexed = allIndexed.filter { it.idx1 >= 0 && it.idx2 >= 0 }

        if (indexed.size != allIndexed.size) {
            Log.w(
                TAG,
                "Skipped ${allIndexed.size - indexed.size} matches with lost keypoint indices"
            )
        }

        Log.i(
            TAG,
            "addPairMatches $frameLeft-$frameRight indexed=${indexed.size}/${matches.size}"
        )

        for (m in indexed) {
            val leftKey = frameLeft to m.idx1
            val rightKey = frameRight to m.idx2

            val leftTrackId = obsToTrack[leftKey]
            val rightTrackId = obsToTrack[rightKey]

            when {
                leftTrackId == null && rightTrackId == null -> {
                    val newId = tracks.size

                    val track = FeatureTrack(
                        id = newId,
                        observations = mutableListOf(
                            FeatureObservation(frameLeft, m.idx1, m.pt1),
                            FeatureObservation(frameRight, m.idx2, m.pt2)
                        )
                    )

                    tracks.add(track)
                    obsToTrack[leftKey] = newId
                    obsToTrack[rightKey] = newId
                }

                leftTrackId != null && rightTrackId == null -> {
                    val track = tracks[leftTrackId]

                    if (track.observations.none { it.frameIndex == frameRight }) {
                        track.observations.add(
                            FeatureObservation(frameRight, m.idx2, m.pt2)
                        )
                        obsToTrack[rightKey] = leftTrackId
                    }
                }

                leftTrackId == null && rightTrackId != null -> {
                    val track = tracks[rightTrackId]

                    if (track.observations.none { it.frameIndex == frameLeft }) {
                        track.observations.add(
                            FeatureObservation(frameLeft, m.idx1, m.pt1)
                        )
                        obsToTrack[leftKey] = rightTrackId
                    }
                }

                leftTrackId == rightTrackId -> {
                    // Already linked. Nothing to do.
                }

                else -> {
                    // Collision: two existing tracks claim this match.
                    // For first version, skip. Do not merge yet.
                    Log.w(
                        TAG,
                        "Track collision on pair $frameLeft-$frameRight: " +
                                "$leftTrackId vs $rightTrackId"
                    )
                }
            }
        }
    }

    fun getTracks(): List<FeatureTrack> {
        return tracks
            .map { track ->
                track.copy(
                    observations = track.observations
                        .sortedBy { it.frameIndex }
                        .toMutableList()
                )
            }
    }

    fun getGoodTracks(minLength: Int = 3): List<FeatureTrack> {
        return getTracks().filter { track ->
            val frames = track.observations.map { it.frameIndex }
            track.observations.size >= minLength &&
                    frames.distinct().size == frames.size
        }
    }

    fun logStats() {
        val hist = tracks
            .groupBy { it.observations.size }
            .mapValues { it.value.size }
            .toSortedMap()

        val good3 = getGoodTracks(3).size
        val good4 = getGoodTracks(4).size

        Log.i(TAG, "tracks total=${tracks.size}")
        Log.i(TAG, "track length histogram=$hist")
        Log.i(TAG, "tracks len>=3=$good3")
        Log.i(TAG, "tracks len>=4=$good4")
    }
}