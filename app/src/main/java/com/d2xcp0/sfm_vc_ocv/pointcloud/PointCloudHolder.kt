package com.d2xcp0.sfm_vc_ocv.pointcloud

import org.opencv.core.Point3

object PointCloudHolder {
    var points: List<Point3> = emptyList()

    // Pontonkenti RGB szin 0..1 tartomanyban (ugyanaz a sorrend, mint points); null = nincs szin
    var colors: List<FloatArray>? = null
}
