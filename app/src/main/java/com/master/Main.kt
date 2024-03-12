package com.master

import android.app.Application
import android.content.Context
import android.os.Build
import android.util.Log
import org.opencv.android.OpenCVLoader
import java.io.File


@Suppress("DEPRECATION")
class Main : Application() {

    override fun onCreate() {
        super.onCreate()
        if (!OpenCVLoader.initDebug())
            Log.e("OpenCV", "Unable to load OpenCV!")
        else
            Log.d("OpenCV", "OpenCV loaded Successfully!")
    }
}

fun getFileFromAssets(context: Context, fileName: String): File = File(context.cacheDir, fileName)
    .also {
        if (!it.exists()) {
            it.outputStream().use { cache ->
                context.assets.open(fileName).use { inputStream ->
                    inputStream.copyTo(cache)
                }
            }
        }
    }