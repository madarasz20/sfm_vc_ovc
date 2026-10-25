package com.d2xcp0.sfm_vc_ocv.sfm

import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point3
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Valodi (egyideju) bundle adjustment: Levenberg-Marquardt az osszes kamerapozon (6 param)
 * es 3D ponton (3 param) egyszerre, minimalizalva a reprojekcios hibat.
 *
 * - Analitikus Jacobi-matrixok, kamera forgas balrol perturbalva: R' = exp([dw]x) * R
 * - Schur-komplemens: a pontok 3x3 blokkjait kiejtve csak a kis kamera-rendszert oldjuk meg
 * - Huber veszteseg (IRLS sulyokkal), hogy a rossz parositasok ne huzzak el a megoldast
 * - Gauge: a fix kamera nem mozog, a masodik kamera tavolsaga rogzitett (skala)
 *
 * Pinhole modell, D = 0 (a kulcspontok mar torzitasmentesek).
 */
class SparseBundleAdjuster(K: Mat) {

    companion object {
        private const val TAG = "SFM_BA"
        private const val HUBER_PX = 1.5
        private const val LAMBDA_INIT = 1e-3
        private const val LAMBDA_MIN = 1e-6
        private const val MAX_LAMBDA_TRIES = 10
        private const val MIN_REL_DECREASE = 1e-6
        private const val BEHIND_PENALTY_PX = 100.0
    }

    data class Stats(
        val initialRms: Double,
        val finalRms: Double,
        val iterations: Int,
        val observations: Int,
        val cameras: Int,
        val points: Int,
        val millis: Long
    )

    private val fx = K.get(0, 0)[0]
    private val fy = K.get(1, 1)[0]
    private val cx = K.get(0, 2)[0]
    private val cy = K.get(1, 2)[0]

    fun adjust(
        rotations: Array<Mat?>,
        translations: Array<Mat?>,
        points: MutableMap<Int, Point3>,
        tracks: List<FeatureTrack>,
        fixedFrame: Int,
        scaleFrame: Int,
        maxIterations: Int
    ): Stats? {
        val start = System.currentTimeMillis()

        // ── Parameterek kigyujtese ───────────────────────────────────────────
        val frames = rotations.indices.filter { rotations[it] != null && translations[it] != null }
        if (frames.size < 2 || points.isEmpty()) return null

        val camOf = HashMap<Int, Int>()
        frames.forEachIndexed { c, f -> camOf[f] = c }
        val nc = frames.size

        val R = Array(nc) { c -> DoubleArray(9) { rotations[frames[c]]!!.get(it / 3, it % 3)[0] } }
        val T = Array(nc) { c -> DoubleArray(3) { translations[frames[c]]!!.get(it, 0)[0] } }

        val fixedCam = camOf[fixedFrame] ?: return null
        val scaleCam = camOf[scaleFrame]
        val scaleNorm = scaleCam?.let { norm3(T[it]) } ?: 0.0

        val ptIds = points.keys.toIntArray()
        val np = ptIds.size
        val ptOf = HashMap<Int, Int>(np * 2)
        ptIds.forEachIndexed { j, id -> ptOf[id] = j }
        val P = DoubleArray(3 * np)
        for (j in 0 until np) {
            val p = points.getValue(ptIds[j])
            P[3 * j] = p.x; P[3 * j + 1] = p.y; P[3 * j + 2] = p.z
        }

        // Megfigyelesek: (kamera, pont, u, v)
        val oCam = ArrayList<Int>()
        val oPt = ArrayList<Int>()
        val oU = ArrayList<Double>()
        val oV = ArrayList<Double>()
        for (track in tracks) {
            val j = ptOf[track.id] ?: continue
            for (o in track.observations) {
                val c = camOf[o.frameIndex] ?: continue
                oCam.add(c); oPt.add(j); oU.add(o.point2D.x); oV.add(o.point2D.y)
            }
        }
        val no = oCam.size
        if (no < 6 * nc) return null

        // Pontonkenti megfigyeles-lista a Schur-komplemenshez
        val obsOfPt = Array(np) { ArrayList<Int>(4) }
        for (k in 0 until no) obsOfPt[oPt[k]].add(k)

        // Szabad kamerak indexe a redukalt rendszerben (-1 = fix)
        val freeIdx = IntArray(nc) { -1 }
        var nFree = 0
        for (c in 0 until nc) if (c != fixedCam) freeIdx[c] = nFree++
        val dim = 6 * nFree

        // Munkatombok
        val Jc = DoubleArray(12 * no)   // 2x6 / megfigyeles
        val Jp = DoubleArray(6 * no)    // 2x3
        val E = DoubleArray(2 * no)     // reziduum (proj - meres)
        val Wt = DoubleArray(no)        // Huber suly
        val Wcp = DoubleArray(18 * no)  // Jc^T w Jp, 6x3

        fun cost(Rs: Array<DoubleArray>, Ts: Array<DoubleArray>, Ps: DoubleArray): Pair<Double, Double> {
            var robust = 0.0
            var sq = 0.0
            for (k in 0 until no) {
                val r = residual(Rs[oCam[k]], Ts[oCam[k]], Ps, oPt[k], oU[k], oV[k])
                if (r == null) {
                    // Kamera moge kerult pont: nagy buntetes, kulonben a "kihagyas" csokkentene a koltseget
                    robust += HUBER_PX * (BEHIND_PENALTY_PX - 0.5 * HUBER_PX)
                    sq += BEHIND_PENALTY_PX * BEHIND_PENALTY_PX
                    continue
                }
                val n2 = r[0] * r[0] + r[1] * r[1]
                sq += n2
                val n = sqrt(n2)
                robust += if (n <= HUBER_PX) 0.5 * n2 else HUBER_PX * (n - 0.5 * HUBER_PX)
            }
            return robust to sqrt(sq / no)
        }

        var (curCost, initialRms) = cost(R, T, P)
        var finalRms = initialRms
        var lambda = LAMBDA_INIT
        var iter = 0

        while (iter < maxIterations) {
            iter++

            // ── Linearizalas ────────────────────────────────────────────────
            val U = Array(nc) { DoubleArray(36) }
            val V = Array(np) { DoubleArray(9) }
            val bc = Array(nc) { DoubleArray(6) }
            val bp = Array(np) { DoubleArray(3) }

            for (k in 0 until no) {
                val c = oCam[k]; val j = oPt[k]
                val Rc = R[c]; val Tc = T[c]
                val X = P[3 * j]; val Y = P[3 * j + 1]; val Z = P[3 * j + 2]

                // RX es kamera koordinatak
                val rx = Rc[0] * X + Rc[1] * Y + Rc[2] * Z
                val ry = Rc[3] * X + Rc[4] * Y + Rc[5] * Z
                val rz = Rc[6] * X + Rc[7] * Y + Rc[8] * Z
                val xc = rx + Tc[0]; val yc = ry + Tc[1]; val zc = rz + Tc[2]

                if (zc <= 1e-9) { Wt[k] = 0.0; continue }

                val iz = 1.0 / zc
                val e0 = fx * xc * iz + cx - oU[k]
                val e1 = fy * yc * iz + cy - oV[k]
                E[2 * k] = e0; E[2 * k + 1] = e1

                val n = sqrt(e0 * e0 + e1 * e1)
                val w = if (n <= HUBER_PX) 1.0 else HUBER_PX / n
                Wt[k] = w

                // d(pi)/d(Xc), 2x3
                val a00 = fx * iz; val a02 = -fx * xc * iz * iz
                val a11 = fy * iz; val a12 = -fy * yc * iz * iz

                // Pont Jacobi: dpi * R
                val jp = 6 * k
                for (col in 0 until 3) {
                    Jp[jp + col] = a00 * Rc[col] + a02 * Rc[6 + col]
                    Jp[jp + 3 + col] = a11 * Rc[3 + col] + a12 * Rc[6 + col]
                }

                // Kamera Jacobi: [dpi * (-[RX]x) | dpi]
                // -[v]x = [[0, vz, -vy], [-vz, 0, vx], [vy, -vx, 0]]
                val jc = 12 * k
                Jc[jc + 0] = a02 * ry
                Jc[jc + 1] = a00 * rz - a02 * rx
                Jc[jc + 2] = -a00 * ry
                Jc[jc + 3] = a00; Jc[jc + 4] = 0.0; Jc[jc + 5] = a02
                Jc[jc + 6] = -a11 * rz + a12 * ry
                Jc[jc + 7] = -a12 * rx
                Jc[jc + 8] = a11 * rx
                Jc[jc + 9] = 0.0; Jc[jc + 10] = a11; Jc[jc + 11] = a12

                // Normalegyenlet blokkok
                val Uc = U[c]; val Vj = V[j]
                for (r in 0 until 6) {
                    val j0 = Jc[jc + r]; val j1 = Jc[jc + 6 + r]
                    for (s in 0 until 6) Uc[6 * r + s] += w * (j0 * Jc[jc + s] + j1 * Jc[jc + 6 + s])
                    bc[c][r] -= w * (j0 * e0 + j1 * e1)
                    for (s in 0 until 3) Wcp[18 * k + 3 * r + s] = w * (j0 * Jp[jp + s] + j1 * Jp[jp + 3 + s])
                }
                for (r in 0 until 3) {
                    val j0 = Jp[jp + r]; val j1 = Jp[jp + 3 + r]
                    for (s in 0 until 3) Vj[3 * r + s] += w * (j0 * Jp[jp + s] + j1 * Jp[jp + 3 + s])
                    bp[j][r] -= w * (j0 * e0 + j1 * e1)
                }
            }

            // Skala rogzitese: a skalakamera transzlacioja ne valtozzon a sajat iranyaban
            val scalePenalty = if (scaleCam != null && scaleCam != fixedCam && scaleNorm > 1e-12) {
                var meanDiag = 0.0
                for (c in 0 until nc) for (r in 0 until 6) meanDiag += U[c][7 * r]
                1e6 * (meanDiag / (6 * nc) + 1.0)
            } else 0.0

            // ── LM lepes probalgatas ────────────────────────────────────────
            var accepted = false
            var tries = 0
            while (!accepted && tries < MAX_LAMBDA_TRIES) {
                tries++

                // V* inverzek
                val Vinv = Array(np) { j ->
                    val m = V[j].copyOf()
                    for (d in 0 until 3) m[4 * d] = m[4 * d] * (1.0 + lambda) + 1e-12
                    inv3(m)
                }

                // Redukalt kamerarendszer S * dc = g
                val S = DoubleArray(dim * dim)
                val g = DoubleArray(dim)

                for (c in 0 until nc) {
                    val fi = freeIdx[c]; if (fi < 0) continue
                    val o = 6 * fi
                    for (r in 0 until 6) {
                        for (s in 0 until 6) S[(o + r) * dim + o + s] += U[c][6 * r + s]
                        S[(o + r) * dim + o + r] += lambda * U[c][7 * r] + 1e-12
                        g[o + r] += bc[c][r]
                    }
                }

                if (scalePenalty > 0.0) {
                    val fi = freeIdx[scaleCam!!]
                    if (fi >= 0) {
                        val tn = T[scaleCam]
                        val o = 6 * fi + 3
                        for (r in 0 until 3) for (s in 0 until 3) {
                            S[(o + r) * dim + o + s] += scalePenalty * tn[r] * tn[s] / (scaleNorm * scaleNorm)
                        }
                    }
                }

                val tmp = DoubleArray(18)   // W * Vinv, 6x3
                for (j in 0 until np) {
                    val Vi = Vinv[j] ?: continue
                    val list = obsOfPt[j]
                    for (ka in list) {
                        if (Wt[ka] == 0.0) continue
                        val fa = freeIdx[oCam[ka]]; if (fa < 0) continue
                        val wa = 18 * ka
                        for (r in 0 until 6) for (s in 0 until 3) {
                            tmp[3 * r + s] = Wcp[wa + 3 * r] * Vi[s] +
                                    Wcp[wa + 3 * r + 1] * Vi[3 + s] +
                                    Wcp[wa + 3 * r + 2] * Vi[6 + s]
                        }
                        val oa = 6 * fa
                        for (r in 0 until 6) {
                            g[oa + r] -= tmp[3 * r] * bp[j][0] + tmp[3 * r + 1] * bp[j][1] + tmp[3 * r + 2] * bp[j][2]
                        }
                        for (kb in list) {
                            if (Wt[kb] == 0.0) continue
                            val fb = freeIdx[oCam[kb]]; if (fb < 0) continue
                            val wb = 18 * kb
                            val ob = 6 * fb
                            for (r in 0 until 6) for (s in 0 until 6) {
                                S[(oa + r) * dim + ob + s] -= tmp[3 * r] * Wcp[wb + 3 * s] +
                                        tmp[3 * r + 1] * Wcp[wb + 3 * s + 1] +
                                        tmp[3 * r + 2] * Wcp[wb + 3 * s + 2]
                            }
                        }
                    }
                }

                val dc = choleskySolve(S, g, dim)
                if (dc == null) {
                    lambda *= 10.0
                    continue
                }

                // Pont lepesek visszahelyettesitessel
                val dP = DoubleArray(3 * np)
                for (j in 0 until np) {
                    val Vi = Vinv[j] ?: continue
                    val rhs = bp[j].copyOf()
                    for (k in obsOfPt[j]) {
                        if (Wt[k] == 0.0) continue
                        val f = freeIdx[oCam[k]]; if (f < 0) continue
                        val o = 6 * f
                        for (s in 0 until 3) {
                            var acc = 0.0
                            for (r in 0 until 6) acc += Wcp[18 * k + 3 * r + s] * dc[o + r]
                            rhs[s] -= acc
                        }
                    }
                    for (r in 0 until 3) dP[3 * j + r] = Vi[3 * r] * rhs[0] + Vi[3 * r + 1] * rhs[1] + Vi[3 * r + 2] * rhs[2]
                }

                // Uj parameterek
                val R2 = Array(nc) { R[it].copyOf() }
                val T2 = Array(nc) { T[it].copyOf() }
                for (c in 0 until nc) {
                    val f = freeIdx[c]; if (f < 0) continue
                    val o = 6 * f
                    R2[c] = mul3(expSO3(dc[o], dc[o + 1], dc[o + 2]), R[c])
                    T2[c][0] += dc[o + 3]; T2[c][1] += dc[o + 4]; T2[c][2] += dc[o + 5]
                }
                if (scaleCam != null && scaleCam != fixedCam && scaleNorm > 1e-12) {
                    val n = norm3(T2[scaleCam])
                    if (n > 1e-12) for (r in 0 until 3) T2[scaleCam][r] *= scaleNorm / n
                }
                val P2 = DoubleArray(3 * np) { P[it] + dP[it] }

                val (newCost, newRms) = cost(R2, T2, P2)
                if (newCost.isFinite() && newCost < curCost) {
                    val rel = (curCost - newCost) / curCost
                    for (c in 0 until nc) { R[c] = R2[c]; T[c] = T2[c] }
                    System.arraycopy(P2, 0, P, 0, P.size)
                    curCost = newCost
                    finalRms = newRms
                    lambda = maxOf(lambda / 3.0, LAMBDA_MIN)
                    accepted = true
                    if (rel < MIN_REL_DECREASE) iter = maxIterations
                } else {
                    lambda *= 5.0
                }
            }

            if (!accepted) break
        }

        // ── Visszairas ─────────────────────────────────────────────────────
        for (c in 0 until nc) {
            val f = frames[c]
            val Rm = Mat(3, 3, CvType.CV_64F)
            for (i in 0 until 9) Rm.put(i / 3, i % 3, R[c][i])
            val Tm = Mat(3, 1, CvType.CV_64F)
            for (i in 0 until 3) Tm.put(i, 0, T[c][i])
            rotations[f] = Rm
            translations[f] = Tm
        }
        for (j in 0 until np) points[ptIds[j]] = Point3(P[3 * j], P[3 * j + 1], P[3 * j + 2])

        val stats = Stats(initialRms, finalRms, iter, no, nc, np, System.currentTimeMillis() - start)
        Log.i(
            TAG,
            "BA: cams=$nc pts=$np obs=$no iters=$iter " +
                    "rms ${"%.3f".format(initialRms)} → ${"%.3f".format(finalRms)} px (${stats.millis} ms)"
        )
        return stats
    }

    // ── Segedfuggvenyek ────────────────────────────────────────────────────

    private fun residual(Rc: DoubleArray, Tc: DoubleArray, P: DoubleArray, j: Int, u: Double, v: Double): DoubleArray? {
        val X = P[3 * j]; val Y = P[3 * j + 1]; val Z = P[3 * j + 2]
        val xc = Rc[0] * X + Rc[1] * Y + Rc[2] * Z + Tc[0]
        val yc = Rc[3] * X + Rc[4] * Y + Rc[5] * Z + Tc[1]
        val zc = Rc[6] * X + Rc[7] * Y + Rc[8] * Z + Tc[2]
        if (zc <= 1e-9) return null
        return doubleArrayOf(fx * xc / zc + cx - u, fy * yc / zc + cy - v)
    }

    private fun norm3(v: DoubleArray) = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    // Rodrigues-formula: exp([w]x)
    private fun expSO3(wx: Double, wy: Double, wz: Double): DoubleArray {
        val th = sqrt(wx * wx + wy * wy + wz * wz)
        if (th < 1e-12) return doubleArrayOf(1.0, -wz, wy, wz, 1.0, -wx, -wy, wx, 1.0)
        val kx = wx / th; val ky = wy / th; val kz = wz / th
        val c = cos(th); val s = sin(th); val v = 1 - c
        return doubleArrayOf(
            c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s,
            ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s,
            kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v
        )
    }

    private fun mul3(a: DoubleArray, b: DoubleArray) = DoubleArray(9) { i ->
        val r = i / 3; val c = i % 3
        a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c]
    }

    private fun inv3(m: DoubleArray): DoubleArray? {
        val det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                m[1] * (m[3] * m[8] - m[5] * m[6]) +
                m[2] * (m[3] * m[7] - m[4] * m[6])
        if (abs(det) < 1e-18) return null
        val id = 1.0 / det
        return doubleArrayOf(
            (m[4] * m[8] - m[5] * m[7]) * id, (m[2] * m[7] - m[1] * m[8]) * id, (m[1] * m[5] - m[2] * m[4]) * id,
            (m[5] * m[6] - m[3] * m[8]) * id, (m[0] * m[8] - m[2] * m[6]) * id, (m[2] * m[3] - m[0] * m[5]) * id,
            (m[3] * m[7] - m[4] * m[6]) * id, (m[1] * m[6] - m[0] * m[7]) * id, (m[0] * m[4] - m[1] * m[3]) * id
        )
    }

    // Szimmetrikus pozitiv definit rendszer megoldasa (Cholesky); null, ha nem PD
    private fun choleskySolve(A: DoubleArray, b: DoubleArray, n: Int): DoubleArray? {
        val L = DoubleArray(n * n)
        for (i in 0 until n) {
            for (j in 0..i) {
                var sum = 0.5 * (A[i * n + j] + A[j * n + i])
                for (k in 0 until j) sum -= L[i * n + k] * L[j * n + k]
                if (i == j) {
                    if (sum <= 0.0 || !sum.isFinite()) return null
                    L[i * n + i] = sqrt(sum)
                } else {
                    L[i * n + j] = sum / L[j * n + j]
                }
            }
        }
        val y = DoubleArray(n)
        for (i in 0 until n) {
            var sum = b[i]
            for (k in 0 until i) sum -= L[i * n + k] * y[k]
            y[i] = sum / L[i * n + i]
        }
        val x = DoubleArray(n)
        for (i in n - 1 downTo 0) {
            var sum = y[i]
            for (k in i + 1 until n) sum -= L[k * n + i] * x[k]
            x[i] = sum / L[i * n + i]
        }
        return x
    }
}
