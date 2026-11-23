
package com.devicethreatdetection.app

import android.content.Context
import org.json.JSONObject
import kotlin.math.abs

data class NetworkScanResult(
    val prediction: String,
    val featureCount: Int,
    val missingFeatureCount: Int
)

class NetworkIntrusionScanner(context: Context) {

    private val featureNames: List<String>
    private val medians: DoubleArray
    private val childrenLeft: IntArray
    private val childrenRight: IntArray
    private val treeFeatures: IntArray
    private val thresholds: DoubleArray
    private val predictions: IntArray

    init {
        val json = context.assets.open("nids_model_android.json")
            .bufferedReader().use { it.readText() }

        val root = JSONObject(json)

        featureNames = root.getJSONArray("feature_names").let { array ->
            List(array.length()) { array.getString(it) }
        }

        val imputer = root.getJSONObject("imputer")
        require(imputer.getString("strategy") == "median")

        medians = imputer.getJSONArray("statistics").toDoubleArray()

        val tree = root.getJSONObject("tree")
        childrenLeft = tree.getJSONArray("children_left").toIntArray()
        childrenRight = tree.getJSONArray("children_right").toIntArray()
        treeFeatures = tree.getJSONArray("feature").toIntArray()
        thresholds = tree.getJSONArray("threshold").toDoubleArray()
        predictions = tree.getJSONArray("predicted_class").toIntArray()

        val count = featureNames.size

        require(count == 78 && medians.size == count) {
            "Unexpected NIDS feature configuration."
        }

        require(
            childrenLeft.size == childrenRight.size &&
                    childrenLeft.size == treeFeatures.size &&
                    childrenLeft.size == thresholds.size &&
                    childrenLeft.size == predictions.size
        ) {
            "Invalid NIDS Decision Tree data."
        }
    }

    fun scanRecord(input: Map<String, Double?>): NetworkScanResult {

        require(input.isNotEmpty()) {
            "No network flow features provided."
        }

        val values = DoubleArray(featureNames.size)
        var missingCount = 0

        for (i in featureNames.indices) {

            val rawValue = input[featureNames[i]]

            val value = rawValue?.takeIf {
                it.isFinite()
            }

            if (value == null) {
                values[i] = medians[i]
                missingCount++
            } else {
                values[i] = value
            }
        }

        var node = 0

        while (childrenLeft[node] != -1) {

            val featureIndex = treeFeatures[node]

            require(featureIndex in values.indices) {
                "Invalid feature index in NIDS tree."
            }

            node = if (
                values[featureIndex] <= thresholds[node]
            ) {
                childrenLeft[node]
            } else {
                childrenRight[node]
            }

            require(node in childrenLeft.indices) {
                "Invalid NIDS tree node."
            }
        }

        val prediction = when (predictions[node]) {
            0 -> "Benign"
            1 -> "Attack"
            else -> throw IllegalStateException(
                "Unexpected NIDS prediction class."
            )
        }

        return NetworkScanResult(
            prediction = prediction,
            featureCount = featureNames.size,
            missingFeatureCount = missingCount
        )
    }

    fun getFeatureNames(): List<String> = featureNames

    private fun org.json.JSONArray.toIntArray(): IntArray =
        IntArray(length()) { getInt(it) }

    private fun org.json.JSONArray.toDoubleArray(): DoubleArray =
        DoubleArray(length()) { getDouble(it) }
}
