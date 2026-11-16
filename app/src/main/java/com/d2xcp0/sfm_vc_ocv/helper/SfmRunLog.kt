package com.d2xcp0.sfm_vc_ocv.helper

import android.os.SystemClock
import android.util.Log

// Egy SfM futas meresi naploja:
// - SFM_TIMING: lepesenkenti ido (a lepes vegen azonnal, a futas vegen egy osszesito sorban)
// - SFM_SUMMARY: egyetlen sor a futas fo szamaival, "; " elvalasztoval (tablazatba masolhato)
class SfmRunLog {

    companion object {
        private const val TAG_TIMING = "SFM_TIMING"
        private const val TAG_SUMMARY = "SFM_SUMMARY"
    }

    private val start = SystemClock.elapsedRealtime()
    private var last = start

    // Fo lepesek: egymast koveto szakaszok, osszeguk ~ a teljes futasido
    private val steps = LinkedHashMap<String, Long>()

    // Reszidok egy fo lepesen belul (pl. a debug kepek mentese a parositason belul)
    private val details = LinkedHashMap<String, Long>()

    private val fields = LinkedHashMap<String, String>()

    private var finished = false

    // Lezarja az elozo lepes ota eltelt szakaszt a megadott neven
    fun step(name: String) {
        val now = SystemClock.elapsedRealtime()
        val ms = now - last
        steps[name] = (steps[name] ?: 0L) + ms
        last = now
        Log.i(TAG_TIMING, "$name: $ms ms")
    }

    // Egy fo lepesen belul kulon mert resz (nem modositja a lepeshatarokat)
    fun detail(name: String, ms: Long) {
        details[name] = (details[name] ?: 0L) + ms
    }

    fun set(key: String, value: Any?) {
        fields[key] = value?.toString() ?: "-"
    }

    // A futas vegen (sikeres vagy sikertelen) egyszer hivando
    fun finish(status: String) {
        if (finished) return
        finished = true

        val total = SystemClock.elapsedRealtime() - start

        val timing = (steps.map { (k, v) -> "$k=$v" } +
                details.map { (k, v) -> "$k=$v" } +
                "total=$total").joinToString("; ")
        Log.i(TAG_TIMING, "all [ms]: $timing")

        val summary = (listOf("status=$status") +
                fields.map { (k, v) -> "$k=$v" } +
                "total_ms=$total").joinToString("; ")
        Log.i(TAG_SUMMARY, summary)
    }
}
