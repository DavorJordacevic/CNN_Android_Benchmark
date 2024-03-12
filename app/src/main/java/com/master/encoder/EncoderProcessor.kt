package com.davor.master.encoder

import android.util.Log
import org.opencv.core.*
import org.opencv.core.Core.norm
import org.opencv.core.Core.transpose
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net

// This is a singleton class (it can have only one instance) user for preprocessing, encoding and comparing the faces
object EncoderProcessor{
    // Default number of faces used for comparison
    private val TARGET_IMG_WIDTH = 112.0
    private val TARGET_IMG_HEIGHT = 112.0
    private val MEAN = Scalar(0.485, 0.456, 0.406)
    lateinit var dnnNet: Net
    private var layerNames = mutableListOf<String>()
    private val desiredKP = Mat(5, 2, CvType.CV_32F)

    init {
        println("Singleton encoder class invoked.")
    }

    // Load the ONNX model and update the network backend
    // TODO Test VULKAN performance on Snapdragon processors
    fun updateBackend(path: String){
        Log.d("ENCODER", "INITIALIZATION")
        dnnNet = Dnn.readNetFromONNX(path)
        dnnNet.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV)
        dnnNet.setPreferableTarget(Dnn.DNN_TARGET_CPU)
        // Set the default layer names and keypoints
        setDefaultValues()
    }

    // Set the default layer names and keypoints
    fun setDefaultValues(){
        layerNames = dnnNet.unconnectedOutLayersNames
        desiredKP.put(0, 0, 38.2946)
        desiredKP.put(0, 1, 59.6963)
        desiredKP.put(1, 0, 73.5318)
        desiredKP.put(1, 1, 59.5014)
        desiredKP.put(2, 0, 56.0252)
        desiredKP.put(2, 1, 79.7366)
        desiredKP.put(3, 0, 41.5493)
        desiredKP.put(3, 1, 100.3655)
        desiredKP.put(4, 0, 70.729904)
        desiredKP.put(4, 1, 100.2041)
    }

    // Preprocess faces
    fun preprocess(img: Mat): Mat {
        // // Create an input blob
        val blob = Dnn.blobFromImage(image, 1.0,
            Size(
                TARGET_IMG_WIDTH,
                TARGET_IMG_HEIGHT
            ),
            MEAN, true, false
        )

        return blob
    }

    // Predict the encodings
    fun predict(img: Mat): Mat {
            // read and process the input face images
            val inputBlob = preprocess(img)
            dnnNet.setInput(inputBlob)
            // inference
            val result = mutableListOf<Mat>()
            dnnNet.forward(result, layerNames)
            val transposed = Mat()
            transpose(result[0], transposed)
        return transposed
    }

    // Compare faces using the cosine distance
    fun compare(vectorA: Mat, vectorB: Mat): Double {
        val cosDist = 1.0 - vectorA.dot(vectorB) / (norm(vectorA) * norm(vectorB))
        Log.d("cosDistEncoder", cosDist.toString())
        return cosDist
    }

}