package com.d2xcp0.sfm_vc_ocv.camera

import android.util.Log

// Egy kepsorozat elso kepenek fokusz- es expozicios ertekei.
// Minden kep uj kamera sessiont nyit (a CameraPreviewScreen bezarul fotozas utan),
// ezert a rogzitett ertekeknek a Camera2CaptureManager-en kivul kell elniuk.
// Fokusz rogzitese nelkul a focus breathing kepenkent kicsit mas fokusztavolsagot ad,
// igy egyetlen K nem lenne pontos minden kepre.
object CaptureLock {

    private const val TAG = "CaptureLock"

    var focusDistance: Float? = null
        private set
    var exposureTimeNs: Long? = null
        private set
    var iso: Int? = null
        private set

    val isLocked: Boolean
        get() = focusDistance != null && exposureTimeNs != null && iso != null

    fun lock(focusDistance: Float, exposureTimeNs: Long, iso: Int) {
        this.focusDistance = focusDistance
        this.exposureTimeNs = exposureTimeNs
        this.iso = iso
        Log.i(TAG, "Locked: focus=$focusDistance diopter, exposure=${exposureTimeNs}ns, iso=$iso")
    }

    // Uj kepsorozat (galeria torlese) eseten
    fun reset() {
        focusDistance = null
        exposureTimeNs = null
        iso = null
        Log.i(TAG, "Reset")
    }
}
