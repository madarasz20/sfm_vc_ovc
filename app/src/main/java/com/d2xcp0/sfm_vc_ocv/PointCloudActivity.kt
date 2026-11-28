package com.d2xcp0.sfm_vc_ocv

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.d2xcp0.sfm_vc_ocv.pointcloud.PointCloudGLSurfaceView

class PointCloudActivity : AppCompatActivity() {

    private lateinit var glView: PointCloudGLSurfaceView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        glView = PointCloudGLSurfaceView(this)
        setContentView(glView)

        // Vissza nyil az ActionBaron (cim: "Point Cloud Viewer", a manifestbol)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    // A vissza nyil ugyanazt teszi, mint a rendszer vissza gombja: vissza a fokepernyore
    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
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
