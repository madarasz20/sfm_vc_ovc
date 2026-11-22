package com.d2xcp0.sfm_vc_ocv.sfm

import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Point3
import java.util.Random
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Szintetikus jelenet a tesztekhez: ismert kamerak egy koriv menten, ismert 3D pontok,
 * es ezek pontos (vagy zajjal terhelt) vetitesei. Igy a tesztelt algoritmusok eredmenye
 * a valodi ertekekkel osszevetheto.
 *
 * Kamerakonvencio: OpenCV (x jobbra, y lefele, z elore), x_kamera = R * X + t.
 */
object SyntheticScene {

    const val FX = 1000.0
    const val FY = 1000.0
    const val CX = 640.0
    const val CY = 480.0

    fun K(): Mat {
        val k = Mat.eye(3, 3, CvType.CV_64F)
        k.put(0, 0, FX); k.put(1, 1, FY)
        k.put(0, 2, CX); k.put(1, 2, CY)
        return k
    }

    // R sorfolytonosan (9 elem), t (3 elem)
    class Cam(val R: DoubleArray, val t: DoubleArray) {
        fun center(): DoubleArray = DoubleArray(3) { c ->
            -(R[c] * t[0] + R[3 + c] * t[1] + R[6 + c] * t[2])
        }
    }

    // Kamera a koriven (XZ sik, sugar r), az origo fele nez
    fun cameraOnArc(angleDeg: Double, radius: Double = 5.0): Cam {
        val a = Math.toRadians(angleDeg)
        val c = doubleArrayOf(radius * sin(a), 0.0, -radius * cos(a))
        val z = normalize(doubleArrayOf(-c[0], -c[1], -c[2]))
        val x = normalize(cross(z, doubleArrayOf(0.0, 1.0, 0.0)))
        val y = cross(z, x)
        val R = doubleArrayOf(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])
        val t = DoubleArray(3) { r -> -(R[3 * r] * c[0] + R[3 * r + 1] * c[1] + R[3 * r + 2] * c[2]) }
        return Cam(R, t)
    }

    fun cameras(angles: List<Double>): List<Cam> = angles.map { cameraOnArc(it) }

    // Veletlen pontok a [-1, 1]^3 kockaban
    fun points(n: Int, rnd: Random): List<Point3> = List(n) {
        Point3(rnd.nextDouble() * 2 - 1, rnd.nextDouble() * 2 - 1, rnd.nextDouble() * 2 - 1)
    }

    fun project(cam: Cam, p: Point3): Point {
        val R = cam.R; val t = cam.t
        val xc = R[0] * p.x + R[1] * p.y + R[2] * p.z + t[0]
        val yc = R[3] * p.x + R[4] * p.y + R[5] * p.z + t[1]
        val zc = R[6] * p.x + R[7] * p.y + R[8] * p.z + t[2]
        return Point(FX * xc / zc + CX, FY * yc / zc + CY)
    }

    // Egy track pontonkent, minden kameraban egy megfigyelessel; noisePx szoras Gauss-zaj
    fun tracks(cams: List<Cam>, pts: List<Point3>, noisePx: Double, rnd: Random): List<FeatureTrack> =
        pts.mapIndexed { j, p ->
            FeatureTrack(
                id = j,
                observations = cams.mapIndexed { i, cam ->
                    val uv = project(cam, p)
                    FeatureObservation(
                        frameIndex = i,
                        keypointIndex = j,
                        point2D = Point(uv.x + noisePx * rnd.nextGaussian(), uv.y + noisePx * rnd.nextGaussian())
                    )
                }.toMutableList()
            )
        }

    fun rMat(R: DoubleArray): Mat {
        val m = Mat(3, 3, CvType.CV_64F)
        for (i in 0 until 9) m.put(i / 3, i % 3, R[i])
        return m
    }

    fun tMat(t: DoubleArray): Mat {
        val m = Mat(3, 1, CvType.CV_64F)
        for (i in 0 until 3) m.put(i, 0, t[i])
        return m
    }

    fun rArray(m: Mat): DoubleArray = DoubleArray(9) { m.get(it / 3, it % 3)[0] }
    fun tArray(m: Mat): DoubleArray = DoubleArray(3) { m.get(it, 0)[0] }

    // Kis veletlen forgatas (fokban megadott szog, veletlen tengely) R ele szorozva
    fun perturbRotation(R: DoubleArray, angleDeg: Double, rnd: Random): DoubleArray {
        val axis = normalize(doubleArrayOf(rnd.nextGaussian(), rnd.nextGaussian(), rnd.nextGaussian()))
        val a = Math.toRadians(angleDeg)
        val rvec = Mat(3, 1, CvType.CV_64F)
        for (i in 0 until 3) rvec.put(i, 0, axis[i] * a)
        val dR = Mat()
        Calib3d.Rodrigues(rvec, dR)
        return mul3(rArray(dR), R)
    }

    // Ket forgatas kozti szog fokban
    fun rotationErrorDeg(A: DoubleArray, B: DoubleArray): Double {
        // trace(A^T B)
        var tr = 0.0
        for (r in 0 until 3) for (c in 0 until 3) tr += A[3 * r + c] * B[3 * r + c]
        return Math.toDegrees(acos(((tr - 1) / 2).coerceIn(-1.0, 1.0)))
    }

    fun dist(a: DoubleArray, b: DoubleArray): Double {
        val dx = a[0] - b[0]; val dy = a[1] - b[1]; val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    fun mul3(a: DoubleArray, b: DoubleArray) = DoubleArray(9) { i ->
        val r = i / 3; val c = i % 3
        a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c]
    }

    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )

    private fun normalize(v: DoubleArray): DoubleArray {
        val n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return doubleArrayOf(v[0] / n, v[1] / n, v[2] / n)
    }
}
