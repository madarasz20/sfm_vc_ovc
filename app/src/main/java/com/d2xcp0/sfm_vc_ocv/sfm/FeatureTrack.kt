package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.core.Point

data class FeatureObservation(
    val frameIndex: Int,
    val keypointIndex: Int,
    val point2D: Point
)

data class FeatureTrack(
    val id: Int,
    val observations: MutableList<FeatureObservation>
)