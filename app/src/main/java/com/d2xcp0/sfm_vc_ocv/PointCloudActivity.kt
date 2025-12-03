package com.d2xcp0.sfm_vc_ocv

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.d2xcp0.sfm_vc_ocv.sfm.PointCloudGLSurfaceView

class PointCloudActivity : AppCompatActivity() {

    private lateinit var glView: PointCloudGLSurfaceView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        glView = PointCloudGLSurfaceView(this)
        setContentView(glView)
    }

    override fun onPause() {
        super.onPause()
        glView.onPause()
    }

    override fun onResume() {
        super.onResume()
        glView.onResume()
    }
}
