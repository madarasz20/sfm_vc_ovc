package com.d2xcp0.sfm_vc_ocv.sfm

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.opencv.core.CvType
import org.opencv.core.Mat

object CalibrationStorage {

    fun save(context: Context, K: Mat, D: Mat) {
        val obj = JSONObject()
        obj.put("K", matToJson(K))
        obj.put("D", matToJson(D))

        val file = context.getFileStreamPath("calibration.json")
        file.writeText(obj.toString())

        android.util.Log.i("CALIB", "Saved calibration.json → ${file.absolutePath}")
        Log.i("CALIB", "Saving to: ${file.absolutePath}")

    }

    fun load(context: Context): Pair<Mat, Mat>? {
        val file = context.getFileStreamPath("calibration.json")
        if (!file.exists()) return null
        Log.i("CALIB", "Loading from: ${file.absolutePath}")

        val text = file.readText()
        val obj = JSONObject(text)

        val K = jsonToMat(obj.getJSONObject("K"))
        val D = jsonToMat(obj.getJSONObject("D"))

        android.util.Log.i("CALIB", "Loaded calibration.json")
        return Pair(K, D)
    }

    private fun matToJson(m: Mat): JSONObject {
        val json = JSONObject()
        json.put("rows", m.rows())
        json.put("cols", m.cols())
        json.put("type", m.type())

        val data = DoubleArray(m.rows() * m.cols())
        m.get(0, 0, data)
        json.put("data", data.joinToString(","))

        return json
    }

    private fun jsonToMat(json: JSONObject): Mat {
        val rows = json.getInt("rows")
        val cols = json.getInt("cols")
        val type = json.getInt("type")

        // Parse comma-separated doubles
        val dataStr = json.getString("data")
        val tokens = dataStr.split(",")

        // Allocate primitive DoubleArray
        val values = DoubleArray(tokens.size)
        for (i in tokens.indices) {
            values[i] = tokens[i].toDouble()
        }

        // Rebuild the matrix
        val m = Mat(rows, cols, type)
        m.put(0, 0, *values)   // <-- IMPORTANT: vararg spread operator
        return m
    }

}
