package com.master

import android.annotation.SuppressLint
import android.app.Activity
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.core.content.ContextCompat
import com.davor.master.encoder.EncoderProcessor
import com.master.master.R
import kotlinx.coroutines.*
import net.vrgsoft.arcprogress.ArcProgressBar
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils


class MainActivity : Activity() {

    lateinit var pBarOpenCV : ArcProgressBar
    lateinit var pBarPytorch : ArcProgressBar
    lateinit var infoOpenCV : TextView
    lateinit var infoPytorch : TextView

    @SuppressLint("SetTextI18n")
    @DelicateCoroutinesApi
    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val filename = "ArcfaceR18"

        val startBtn : Button = findViewById(R.id.startBtn)

        pBarOpenCV = findViewById(R.id.pBarOpenCV)
        pBarOpenCV.visibility = View.VISIBLE

        pBarPytorch = findViewById(R.id.pBarPytorch)
        pBarPytorch.visibility = View.VISIBLE

        val textViewOpenCV : TextView = findViewById(R.id.textViewOpenCV)
        infoOpenCV = findViewById(R.id.infoOpenCV)
        textViewOpenCV.text = filename + " via " + textViewOpenCV.text

        val textViewPytorch : TextView = findViewById(R.id.textViewPytorch)
        infoPytorch = findViewById(R.id.infoPytorch)
        textViewPytorch.text = filename + " via " + textViewPytorch.text

        startBtn.setOnClickListener {
            startBtn.isFocusable = false
            startBtn.isEnabled = false
            startBtn.setTextColor(ContextCompat.getColor(startBtn.context, R.color.gray))

            pBarOpenCV.progress = 0
            pBarPytorch.progress = 0

            GlobalScope.launch {
                benchmarkPytorch(filename)
                benchmarkOpenCV(filename)

                runOnUiThread {
                    startBtn.isFocusable = true
                    startBtn.isEnabled = true
                    startBtn.setTextColor(ContextCompat.getColor(startBtn.context, R.color.white))
                }
            }
        }
    }

    @SuppressLint("SetTextI18n")
    private suspend fun benchmarkOpenCV(filename: String) {
            val model_path = getFileFromAssets(this@MainActivity, "$filename.onnx").absolutePath

            val encoder = EncoderProcessor
            encoder.updateBackend(model_path)

            var elapsedTimeOnnx : Float = 0.0F
            val image: Mat = Utils.loadResource(this@MainActivity, R.raw.image)
            var j = pBarOpenCV.progress

            for (i in 1..100) {
                val begin = System.currentTimeMillis()
                val vector = encoder.predict(image)
                val end = System.currentTimeMillis()
                elapsedTimeOnnx += (end - begin)
                j += 1
                pBarOpenCV.progress = j
            }
            val averageTimeOnnx = elapsedTimeOnnx / 100
            runOnUiThread {
                infoOpenCV.text = "Average inference speed: $averageTimeOnnx ms"
            }
            Log.d("MainActivity", "Elapsed time in milliseconds for OpenCV Mobile is: $averageTimeOnnx milliseconds.")
    }

    @SuppressLint("SetTextI18n")
    private suspend fun benchmarkPytorch(filename: String) {
        withContext(Dispatchers.Default) {
            val model_path = getFileFromAssets(this@MainActivity, "$filename.ptl").absolutePath
            val module: Module = LiteModuleLoader.load(model_path)

            val bitmap = BitmapFactory.decodeStream(assets.open("image.jpg"))
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            var elapsedTimePytorch : Float = 0.0F
            var j = pBarPytorch.progress
            for (i in 1..100) {
                val begin = System.currentTimeMillis()
                val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
                val vectorFloat = outputTensor.dataAsFloatArray
                val end = System.currentTimeMillis()
                elapsedTimePytorch += (end - begin)
                j += 1
                pBarPytorch.progress = j
            }
            val averageTimePytorch = elapsedTimePytorch / 100
            runOnUiThread {
                infoPytorch.text = "Average inference speed: $averageTimePytorch ms"
            }
            Log.d("MainActivity", "Elapsed time in milliseconds for Pytorch Mobile is: $averageTimePytorch milliseconds.")
        }
    }
}
