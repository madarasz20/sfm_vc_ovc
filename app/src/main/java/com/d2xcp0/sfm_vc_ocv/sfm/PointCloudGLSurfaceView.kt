package com.d2xcp0.sfm_vc_ocv.sfm

import android.content.Context
import android.opengl.GLSurfaceView
import android.view.MotionEvent

class PointCloudGLSurfaceView(context: Context) : GLSurfaceView(context) {

    private val renderer: PointCloudRenderer

    init {
        setEGLContextClientVersion(2)
        renderer = PointCloudRenderer()
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    private var previousX = 0f
    private var previousY = 0f

    override fun onTouchEvent(e: MotionEvent): Boolean {
        val x = e.x
        val y = e.y

        when (e.action) {
            MotionEvent.ACTION_MOVE -> {
                val dx = x - previousX
                val dy = y - previousY

                renderer.angleY += dx * 0.5f
                renderer.angleX += dy * 0.5f
            }
        }

        previousX = x
        previousY = y
        return true
    }
}
